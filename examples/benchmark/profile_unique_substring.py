"""Profile runtime-bound unique substrings for coding-agent search-and-replace calls."""

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from profile_ablations import ROOT, cmd, describe, environment, sha, snap, utc, write_json

SIZES = [10 * 1024, 100 * 1024, 1024 * 1024]
CORPORA = ["unique-heavy", "repetitive"]
VARIANTS = ["runtime-cold", "runtime-warm", "static-compiled", "posthoc"]


def overlapping_count(source: bytes, candidate: bytes) -> int:
    if not candidate:
        return len(source) + 1
    count, offset = 0, 0
    while True:
        offset = source.find(candidate, offset)
        if offset == -1:
            return count
        count += 1
        offset += 1


def fit_source(size: int, corpus: str, marker: bytes) -> bytes:
    if corpus == "unique-heavy":
        lines = []
        index = 0
        total = 0
        while total < size:
            line = (
                f"def handler_{index:07d}(request):\n"
                f"    value_{index:07d} = request.get('field_{index:07d}')\n"
                f"    return value_{index:07d}\n"
            ).encode()
            lines.append(line)
            total += len(line)
            index += 1
        source = b"".join(lines)[:size]
    else:
        block = (
            b"def shared_handler(request):\n"
            b"    shared_value = request.get('field')\n"
            b"    return shared_value\n"
        )
        source = (block * (size // len(block) + 1))[:size]
    position = size // 2
    source = source[:position] + marker + source[position + len(marker) :]
    assert len(source) == size and overlapping_count(source, marker) == 1
    return source


def prepare(args):
    from transformers import AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer.save_pretrained(args.output / "tokenizer")
    sources = args.output / "sources"
    sources.mkdir()
    workloads = []
    for size in SIZES:
        for corpus in CORPORA:
            name = f"{corpus}-{size // 1024}k"
            marker = f"UNIQUE_REPLACEMENT_TARGET_{corpus}_{size}_a7f3".encode()
            source = fit_source(size, corpus, marker)
            path = sources / f"{name}.txt"
            path.write_bytes(source)
            ambiguous = b"return shared_value" if corpus == "repetitive" else b"    return value_"
            assert overlapping_count(source, ambiguous) > 1
            token_ids = tokenizer.encode(marker.decode(), add_special_tokens=False)
            workloads.append(
                {
                    "name": name,
                    "corpus": corpus,
                    "source_bytes": size,
                    "source_file": str(path.relative_to(args.output)),
                    "source_sha256": sha(source),
                    "unique_target": marker.decode(),
                    "unique_target_occurrences": overlapping_count(source, marker),
                    "ambiguous_target": ambiguous.decode(),
                    "ambiguous_target_occurrences": overlapping_count(source, ambiguous),
                    "missing_target": "THIS_TARGET_DOES_NOT_EXIST_91c4",
                    "token_ids": token_ids,
                }
            )
    write_json(args.output / "workloads.json", workloads)
    write_json(
        args.output / "metadata.json",
        {
            "status": "prepared",
            "arguments": {
                **vars(args),
                "output": str(args.output),
                "tokenizer": str(args.tokenizer),
            },
            "environment": environment(),
            "source_commit": cmd("git", "rev-parse", "HEAD"),
            "git_status": cmd("git", "status", "--short"),
            "script_sha256": sha(Path(__file__).read_bytes()),
            "tokenizer_files": {
                p.name: sha(p.read_bytes()) for p in (args.output / "tokenizer").iterdir()
            },
            "description": (
                "Synthetic coding-agent old_str benchmark; GPT-2 vocabulary; CPU only; "
                "runtime-bound suffix index versus static grammar compilation and post-hoc validation."
            ),
        },
    )
    shutil.copyfile(__file__, args.output / "benchmark_script.py")
    (args.output / "source.diff").write_text(cmd("git", "diff", "HEAD")["stdout"])


def mask_allows(mask, token_id):
    word, bit = divmod(token_id, 32)
    return bool((int(mask[0, word].item()) & 0xFFFFFFFF) & (1 << bit))


def runtime_correctness(workload, source, info, eos_id, xgr, seed):
    matcher = xgr.UniqueSubstringMatcher(source, info)
    mask = xgr.allocate_token_bitmask(1, info.vocab_size)
    prefix = b""
    rng = random.Random(seed)
    decoded_vocab = info.decoded_vocab
    sampled_ids = set(rng.sample(range(info.vocab_size), min(32, info.vocab_size)))
    sampled_ids.update(workload["token_ids"])
    sampled_ids.add(eos_id)
    checked = 0
    for selected in workload["token_ids"]:
        matcher.fill_next_token_bitmask(mask)
        for token_id in sampled_ids:
            piece = decoded_vocab[token_id]
            expected = (
                overlapping_count(source, prefix) == 1
                if token_id == eos_id
                else bool(piece) and source.find(prefix + piece) != -1
            )
            assert mask_allows(mask, token_id) == expected, (workload["name"], prefix, token_id)
            checked += 1
        assert matcher.accept_token(selected)
        prefix += decoded_vocab[selected]
    assert prefix == workload["unique_target"].encode()
    assert matcher.is_completed and matcher.occurrence_count == 1
    matcher.fill_next_token_bitmask(mask)
    assert mask_allows(mask, eos_id)
    assert matcher.accept_token(eos_id) and matcher.is_terminated

    ambiguous = xgr.UniqueSubstringMatcher(source, info)
    assert ambiguous.accept_string(workload["ambiguous_target"].encode())
    assert ambiguous.occurrence_count == workload["ambiguous_target_occurrences"]
    assert not ambiguous.is_completed and not ambiguous.accept_token(eos_id)
    missing = xgr.UniqueSubstringMatcher(source, info)
    assert not missing.accept_string(workload["missing_target"].encode())
    return {"sampled_mask_checks": checked, "passed": True}


def replay_tokens(matcher, mask, replay_token_ids):
    mask_ns, accept_ns = [], []
    for token_id in replay_token_ids:
        start = time.perf_counter_ns()
        matcher.fill_next_token_bitmask(mask)
        mask_ns.append(time.perf_counter_ns() - start)
        start = time.perf_counter_ns()
        assert matcher.accept_token(token_id)
        accept_ns.append(time.perf_counter_ns() - start)
    return mask_ns, accept_ns


def runtime_replay(source, replay_token_ids, info, xgr, mask, matcher=None):
    setup_start = time.perf_counter_ns()
    if matcher is None:
        matcher = xgr.UniqueSubstringMatcher(source, info)
    else:
        matcher.reset()
    setup_ns = time.perf_counter_ns() - setup_start
    mask_ns, accept_ns = replay_tokens(matcher, mask, replay_token_ids)
    assert matcher.is_terminated
    return {
        "setup_wall_ns": setup_ns,
        "mask_ns_by_token": mask_ns,
        "accept_ns_by_token": accept_ns,
        "mask_total_ns": sum(mask_ns),
        "accept_total_ns": sum(accept_ns),
        "tokens": len(mask_ns),
        "index_states": matcher.num_index_states,
    }, matcher


def static_replay(source, replay_token_ids, info, xgr, mask):
    start = time.perf_counter_ns()
    grammar = xgr.Grammar.from_substring(source, unique=True)
    compiler = xgr.GrammarCompiler(info, cache_enabled=False, max_threads=1)
    compiled = compiler.compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled)
    setup_ns = time.perf_counter_ns() - start
    mask_ns, accept_ns = replay_tokens(matcher, mask, replay_token_ids)
    assert matcher.is_terminated()
    return {
        "setup_wall_ns": setup_ns,
        "mask_ns_by_token": mask_ns,
        "accept_ns_by_token": accept_ns,
        "mask_total_ns": sum(mask_ns),
        "accept_total_ns": sum(accept_ns),
        "tokens": len(mask_ns),
    }, compiled


def posthoc_replay(source, target):
    start = time.perf_counter_ns()
    count = overlapping_count(source, target)
    validation_ns = time.perf_counter_ns() - start
    assert count == 1
    return {"validation_wall_ns": validation_ns, "occurrences": count, "tokens": 0}


def worker(args):
    import psutil
    import torch
    import xgrammar as xgr
    from transformers import AutoTokenizer

    torch.set_num_threads(1)
    directory = args.output / f"{args.variant}-round-{args.round}"
    directory.mkdir()
    tokenizer = AutoTokenizer.from_pretrained(args.output / "tokenizer", local_files_only=True)
    info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    eos_id = tokenizer.eos_token_id
    workloads = json.loads((args.output / "workloads.json").read_text())
    random.Random(args.seed + args.round).shuffle(workloads)
    process = psutil.Process()
    metadata = {
        "status": "running",
        "variant": args.variant,
        "round": args.round,
        "environment": environment(),
        "package_path": xgr.__file__,
        "library_path": str(Path(xgr.__file__).parent / "libxgrammar_bindings.dylib"),
        "timer": vars(time.get_clock_info("perf_counter")),
        "timed_scope": (
            "Total timing surrounds setup/reset and the full token replay. Per-token mask and "
            "accept calls are also recorded separately. "
            "Static setup includes grammar, compiler, and matcher construction. Post-hoc timing includes "
            "only overlapping occurrence validation. Buffer allocation, memory and process telemetry, "
            "and logging are excluded."
        ),
    }
    write_json(directory / "metadata.json", metadata)
    checks = []
    try:
        with (directory / "iterations.jsonl").open("w", buffering=1) as log:
            for workload in workloads:
                source = (args.output / workload["source_file"]).read_bytes()
                checks.append(
                    {
                        "workload": workload["name"],
                        **runtime_correctness(workload, source, info, eos_id, xgr, args.seed),
                    }
                )
                write_json(directory / "correctness.json", checks)
                if args.variant == "static-compiled" and len(source) > args.static_max_bytes:
                    continue
                warm_matcher = (
                    xgr.UniqueSubstringMatcher(source, info)
                    if args.variant == "runtime-warm"
                    else None
                )
                mask = xgr.allocate_token_bitmask(1, info.vocab_size)
                replay_token_ids = (*workload["token_ids"], eos_id)
                posthoc_target = workload["unique_target"].encode()
                for phase, count in (("warmup", args.warmups), ("timed", args.iterations)):
                    for iteration in range(count):
                        before = snap(process)
                        cpu_start = time.process_time_ns()
                        wall_start = time.perf_counter_ns()
                        measured_object = None
                        if args.variant.startswith("runtime"):
                            measurement, measured_object = runtime_replay(
                                source, replay_token_ids, info, xgr, mask, warm_matcher
                            )
                        elif args.variant == "static-compiled":
                            measurement, measured_object = static_replay(
                                source, replay_token_ids, info, xgr, mask
                            )
                        else:
                            measurement = posthoc_replay(source, posthoc_target)
                        total_wall_ns = time.perf_counter_ns() - wall_start
                        total_cpu_ns = time.process_time_ns() - cpu_start
                        if args.variant.startswith("runtime"):
                            measurement["constraint_memory_bytes"] = (
                                measured_object.memory_size_bytes
                            )
                        elif args.variant == "static-compiled":
                            measurement["compiled_memory_bytes"] = measured_object.memory_size_bytes
                        record = {
                            "experiment": "unique-substring",
                            "variant": args.variant,
                            "round": args.round,
                            "phase": phase,
                            "iteration": iteration,
                            "workload": workload["name"],
                            "corpus": workload["corpus"],
                            "source_bytes": len(source),
                            "source_sha256": workload["source_sha256"],
                            "target_bytes": len(workload["unique_target"].encode()),
                            "target_tokens": len(workload["token_ids"]),
                            "vocab_size": info.vocab_size,
                            "cache_enabled": False,
                            "max_threads": 1,
                            "started_utc": utc(),
                            "before": before,
                            "after": snap(process),
                            "total_wall_ns": total_wall_ns,
                            "total_cpu_ns": total_cpu_ns,
                            **measurement,
                            "status": "ok",
                        }
                        log.write(json.dumps(record) + "\n")
                print(
                    f"{args.variant} round {args.round + 1}: {workload['name']} complete",
                    flush=True,
                )
        metadata.update(status="complete", finished_environment=environment())
    except BaseException as exc:
        metadata.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(directory / "metadata.json", metadata)


def summarize(args):
    rows = []
    for round_index in range(args.rounds):
        for variant in VARIANTS:
            directory = args.output / f"{variant}-round-{round_index}"
            assert json.loads((directory / "metadata.json").read_text())["status"] == "complete"
            rows.extend(
                json.loads(line)
                for line in (directory / "iterations.jsonl").read_text().splitlines()
            )
    summary = []
    workload_names = sorted({row["workload"] for row in rows})
    for workload in workload_names:
        for variant in VARIANTS:
            subset = [
                row
                for row in rows
                if row["workload"] == workload
                and row["variant"] == variant
                and row["phase"] == "timed"
            ]
            if not subset:
                continue
            item = {
                "workload": workload,
                "variant": variant,
                "samples": len(subset),
                "total_ms": describe([row["total_wall_ns"] / 1e6 for row in subset]),
            }
            if variant.startswith("runtime") or variant == "static-compiled":
                item.update(
                    setup_ms=describe([row["setup_wall_ns"] / 1e6 for row in subset]),
                    mask_us_per_token=describe(
                        [row["mask_total_ns"] / row["tokens"] / 1000 for row in subset]
                    ),
                    accept_us_per_token=describe(
                        [row["accept_total_ns"] / row["tokens"] / 1000 for row in subset]
                    ),
                )
            else:
                item["validation_us"] = describe(
                    [row["validation_wall_ns"] / 1000 for row in subset]
                )
            summary.append(item)
    write_json(args.output / "summary.json", summary)
    write_json(
        args.output / "verification.json",
        {
            "passed": True,
            "records": len(rows),
            "timed_records": sum(row["phase"] == "timed" for row in rows),
            "expected_samples_per_configuration": args.iterations * args.rounds,
        },
    )
    lines = [
        "# Runtime unique-substring profiling",
        "",
        f"Each configuration used {args.warmups} warmups and {args.iterations} timed repetitions "
        f"in each of {args.rounds} rounds. Values below are pooled medians.",
        "",
        "| Workload | Variant | Setup ms | Mask µs/token | Total ms |",
        "|---|---|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['workload']} | {item['variant']} | "
            f"{item.get('setup_ms', {}).get('median', float('nan')):.3f} | "
            f"{item.get('mask_us_per_token', {}).get('median', float('nan')):.3f} | "
            f"{item['total_ms']['median']:.3f} |"
        )
    lines += [
        "",
        "`runtime-cold` rebuilds the suffix index and lazily computes masks for every replay. "
        "`runtime-warm` reuses one bound index and its realized state masks, resetting only decode "
        "state. `static-compiled` rebuilds a Grammar and CompiledGrammar and is limited to sources "
        f"at most {args.static_max_bytes} bytes. `posthoc` validates only after generation and does "
        "not prevent invalid output, so its time is not an equivalent constrained-decoding baseline.",
        "",
        "Raw setup, per-token mask, acceptance, memory, process telemetry, inputs, hashes, and "
        "correctness samples are stored with the results.",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    metadata = json.loads((args.output / "metadata.json").read_text())
    metadata.update(status="complete", finished_utc=utc())
    write_json(args.output / "metadata.json", metadata)
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "profiling-results/full/tokenizer")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--static-max-bytes", type=int, default=10 * 1024)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.tokenizer = args.tokenizer.resolve()
    assert min(args.warmups, args.iterations, args.rounds, args.static_max_bytes) > 0
    if args.worker:
        worker(args)
        return
    prepare(args)
    for round_index in range(args.rounds):
        order = VARIANTS if round_index % 2 == 0 else list(reversed(VARIANTS))
        for variant in order:
            env = dict(
                os.environ,
                PYTHONPATH=str(ROOT / "python"),
                TOKENIZERS_PARALLELISM="false",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--output",
                    str(args.output),
                    "--tokenizer",
                    str(args.tokenizer),
                    "--variant",
                    variant,
                    "--round",
                    str(round_index),
                    "--warmups",
                    str(args.warmups),
                    "--iterations",
                    str(args.iterations),
                    "--rounds",
                    str(args.rounds),
                    "--static-max-bytes",
                    str(args.static_max_bytes),
                    "--seed",
                    str(args.seed),
                ],
                env=env,
                check=True,
            )
    summarize(args)


if __name__ == "__main__":
    main()
