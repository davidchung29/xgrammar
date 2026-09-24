"""Profile explicit TagDispatch against an equivalent ordinary-Earley grammar."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from profile_ablations import ROOT, cmd, describe, environment, replay, sha, snap, utc, write_json


VARIANTS = ["ordinary-ebnf", "tag-dispatch"]
TAG_COUNTS = [10, 50, 100]


def quote(text):
    return json.dumps(text, ensure_ascii=True)


def make_tags(count):
    return [f"<function=tool_{i:03d}>" for i in range(count)]


def make_bodies(count):
    return "\n".join(f'body_{i:03d} ::= "{{" "\\\"value\\\"" ":" [0-9]+ "}}"' for i in range(count))


def make_tag_dispatch_grammar(count):
    pairs = ",\n  ".join(f"({quote(tag)}, body_{i:03d})" for i, tag in enumerate(make_tags(count)))
    return f"root ::= TagDispatch(\n  {pairs},\n  loop_after_dispatch=false\n)\n{make_bodies(count)}\n"


def longest_prefix_suffix(value, prefixes):
    for length in range(min(len(value), max(map(len, prefixes))), -1, -1):
        suffix = value[-length:] if length else ""
        if suffix in prefixes:
            return suffix
    raise AssertionError(value)


def make_ordinary_grammar(count):
    """Generate a right-linear DFA grammar for text ending at the first complete tool tag."""
    tags = make_tags(count)
    completed = {tag: i for i, tag in enumerate(tags)}
    prefixes = {""}
    for tag in tags:
        prefixes.update(tag[:i] for i in range(1, len(tag)))
    ordered = sorted(prefixes, key=lambda x: (len(x), x))
    state_id = {prefix: i for i, prefix in enumerate(ordered)}
    alphabet = sorted(set("".join(tags)))
    # None of the generated tag characters require EBNF character-class escaping.
    assert not (set(alphabet) & set("\\]^-"))
    complement = "[^" + "".join(alphabet) + "]"
    rules = []
    for prefix in ordered:
        # TagDispatch permits generation to end without seeing a trigger, including
        # while a trigger prefix is buffered as ordinary text.
        alternatives = ['""']
        for char in alphabet:
            candidate = prefix + char
            if candidate in completed:
                target = f"body_{completed[candidate]:03d}"
            else:
                target_prefix = longest_prefix_suffix(candidate, prefixes)
                target = f"scan_{state_id[target_prefix]}"
            alternatives.append(f"{quote(char)} {target}")
        alternatives.append(f"{complement} scan_0")
        rules.append(f"scan_{state_id[prefix]} ::= " + " | ".join(alternatives))
    return "root ::= scan_0\n" + "\n".join(rules) + "\n" + make_bodies(count) + "\n"


def mask_trace(compiled, token_ids, invalid_utf8_ids, xgr):
    import numpy as np

    matcher = xgr.GrammarMatcher(compiled)
    vocab_size = compiled.tokenizer_info.vocab_size
    mask = xgr.allocate_token_bitmask(1, vocab_size)
    hashes = []
    full_hashes = []
    invalid_allowed_counts = []
    for i, token in enumerate(token_ids):
        mask.fill_(-1)
        if not matcher.fill_next_token_bitmask(mask):
            mask.fill_(-1)
        if vocab_size % 32:
            mask[0, -1] &= (1 << (vocab_size % 32)) - 1
        raw = mask.numpy().copy()
        full_hashes.append(sha(raw.tobytes()))
        canonical = raw.view(np.uint32)
        invalid_allowed = 0
        for token_id in invalid_utf8_ids:
            word, bit = divmod(token_id, 32)
            if (int(canonical[0, word]) >> bit) & 1:
                invalid_allowed += 1
            canonical[0, word] &= ~(np.uint32(1) << np.uint32(bit))
        hashes.append(sha(raw.tobytes()))
        invalid_allowed_counts.append(invalid_allowed)
        assert matcher.accept_token(token), (i, token)
    assert matcher.is_terminated()
    return {
        "valid_utf8_prefix_mask_sha256": hashes,
        "full_prefix_mask_sha256": full_hashes,
        "invalid_utf8_allowed_counts": invalid_allowed_counts,
        "invalid_utf8_token_count": len(invalid_utf8_ids),
        "prefixes": len(token_ids),
        "terminated": True,
    }


def prepare(args):
    from transformers import AutoTokenizer

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "profiling-results/full/tokenizer", local_files_only=True)
    tokenizer.save_pretrained(out / "tokenizer")
    workloads = []
    for count in TAG_COUNTS:
        selected = count // 2
        completion = f"The requested tool follows. <function=tool_{selected:03d}>{{\"value\":42}}"
        variants = {
            "ordinary-ebnf": make_ordinary_grammar(count),
            "tag-dispatch": make_tag_dispatch_grammar(count),
        }
        workloads.append({
            "name": f"tags_{count}",
            "tag_count": count,
            "selected_tool": selected,
            "completion": completion,
            "token_ids": tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.eos_token_id],
            "grammars": variants,
            "grammar_sha256": {name: sha(text.encode()) for name, text in variants.items()},
        })
    write_json(out / "workloads.json", workloads)
    for workload in workloads:
        for variant, grammar in workload["grammars"].items():
            (out / f"{workload['name']}-{variant}.ebnf").write_text(grammar)
    package = ROOT / "profiling-builds/compressed/package"
    binary_hashes = {str(p): sha(p.read_bytes()) for p in package.glob("xgrammar/*.dylib")}
    write_json(out / "metadata.json", {
        "status": "prepared",
        "arguments": {**vars(args), "output": str(out)},
        "environment": environment(),
        "source_commit": cmd("git", "rev-parse", "HEAD"),
        "git_status": cmd("git", "status", "--short"),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "helper_sha256": sha((ROOT / "examples/benchmark/profile_ablations.py").read_bytes()),
        "package": str(package),
        "binary_hashes": binary_hashes,
        "tokenizer_files": {p.name: sha(p.read_bytes()) for p in (out / "tokenizer").iterdir()},
        "description": "Explicit TagDispatch versus an equivalent generated right-linear DFA expressed as ordinary EBNF. No LLM or GPU.",
    })
    shutil.copyfile(__file__, out / "benchmark_script.py")
    shutil.copyfile(ROOT / "examples/benchmark/profile_ablations.py", out / "profile_ablations.py")
    (out / "source.diff").write_text(cmd("git", "diff", "HEAD")["stdout"])


def worker(args):
    import psutil
    import torch
    import xgrammar as xgr
    from transformers import AutoTokenizer

    torch.set_num_threads(1)
    out = args.output / f"{args.variant}-round-{args.round}"
    out.mkdir()
    package = ROOT / "profiling-builds/compressed/package"
    assert Path(xgr.__file__).is_relative_to(package)
    meta = {
        "status": "running", "variant": args.variant, "round": args.round,
        "environment": environment(), "package_path": xgr.__file__,
        "binary_hashes": {str(p): sha(p.read_bytes()) for p in Path(xgr.__file__).parent.glob("*.dylib")},
        "timer": vars(time.get_clock_info("perf_counter")),
        "timed_scope": "Fresh-compiler grammar compilation, then per-token mask and acceptance calls. Grammar creation, matcher creation, allocation, telemetry, hashing, and logging are excluded from their respective timers.",
    }
    write_json(out / "metadata.json", meta)
    tokenizer = AutoTokenizer.from_pretrained(args.output / "tokenizer", local_files_only=True)
    info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    invalid_utf8_ids = []
    for token_id, token_bytes in enumerate(info.decoded_vocab):
        try:
            token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            invalid_utf8_ids.append(token_id)
    workloads = json.loads((args.output / "workloads.json").read_text())
    random.Random(args.seed + args.round).shuffle(workloads)
    process = psutil.Process()
    checks = []
    try:
        with (out / "iterations.jsonl").open("w", buffering=1) as log:
            for workload in workloads:
                grammar_text = workload["grammars"][args.variant]
                grammar = xgr.Grammar.from_ebnf(grammar_text)
                printed = str(grammar)
                contains_tag_dispatch = "TagDispatch" in printed
                assert contains_tag_dispatch == (args.variant == "tag-dispatch")
                # One untimed compilation supplies a full-mask correctness trace.
                check_compiler = xgr.GrammarCompiler(info, cache_enabled=False, max_threads=1)
                check_compiled = check_compiler.compile_grammar(grammar)
                checks.append({
                    "workload": workload["name"],
                    "variant": args.variant,
                    "contains_tag_dispatch": contains_tag_dispatch,
                    "compiled_bytes": check_compiled.memory_size_bytes,
                    **mask_trace(check_compiled, workload["token_ids"], invalid_utf8_ids, xgr),
                })
                write_json(out / "correctness.json", checks)
                for phase, count in [("warmup", args.warmups), ("timed", args.iterations)]:
                    for iteration in range(count):
                        before = snap(process)
                        compiler = xgr.GrammarCompiler(info, cache_enabled=False, max_threads=1)
                        cpu_start = time.process_time_ns()
                        start = time.perf_counter_ns()
                        compiled = compiler.compile_grammar(grammar)
                        compile_wall_ns = time.perf_counter_ns() - start
                        compile_cpu_ns = time.process_time_ns() - cpu_start
                        timing = replay(compiled, workload["token_ids"], xgr)
                        entry = {
                            "experiment": "tag-dispatch", "variant": args.variant,
                            "round": args.round, "workload": workload["name"],
                            "tag_count": workload["tag_count"], "phase": phase,
                            "iteration": iteration, "started_utc": utc(),
                            "before": before, "after": snap(process),
                            "cache_enabled": False, "max_threads": 1,
                            "vocab_size": info.vocab_size,
                            "grammar_sha256": workload["grammar_sha256"][args.variant],
                            "compile_wall_ns": compile_wall_ns,
                            "compile_cpu_ns": compile_cpu_ns,
                            "compiled_bytes": compiled.memory_size_bytes,
                            **timing, "status": "ok",
                        }
                        log.write(json.dumps(entry) + "\n")
                print(f"{args.variant} round {args.round + 1}: {workload['name']} complete", flush=True)
        meta.update(status="complete", finished_environment=environment())
    except BaseException as exc:
        meta.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(out / "metadata.json", meta)


def summarize(args):
    rows = []
    equivalence = []
    names = [f"tags_{count}" for count in TAG_COUNTS]
    for round_index in range(args.rounds):
        traces = {}
        for variant in VARIANTS:
            directory = args.output / f"{variant}-round-{round_index}"
            assert json.loads((directory / "metadata.json").read_text())["status"] == "complete"
            traces[variant] = {x["workload"]: x for x in json.loads((directory / "correctness.json").read_text())}
            records = [json.loads(line) for line in (directory / "iterations.jsonl").read_text().splitlines()]
            for name in names:
                for phase, count in [("warmup", args.warmups), ("timed", args.iterations)]:
                    assert sum(x["workload"] == name and x["phase"] == phase for x in records) == count
            for record in records:
                assert len(record["mask_ns_by_token"]) == len(record["accept_ns_by_token"]) == record["tokens"]
                assert sum(record["mask_ns_by_token"]) == record["mask_total_ns"]
            rows.extend(records)
        for name in names:
            left, right = traces["ordinary-ebnf"][name], traces["tag-dispatch"][name]
            assert left["valid_utf8_prefix_mask_sha256"] == right["valid_utf8_prefix_mask_sha256"], (round_index, name)
            full_mismatches = sum(
                a != b for a, b in zip(left["full_prefix_mask_sha256"], right["full_prefix_mask_sha256"])
            )
            equivalence.append({
                "round": round_index, "workload": name, "passed": True,
                "scope": "all GPT-2 tokens whose individual decoded byte piece is valid UTF-8, plus EOS",
                "prefixes_compared": left["prefixes"],
                "invalid_utf8_token_count": left["invalid_utf8_token_count"],
                "full_mask_mismatched_prefixes": full_mismatches,
            })
    summary = []
    for name in names:
        for variant in VARIANTS:
            for round_index in [*range(args.rounds), "all"]:
                subset = [x for x in rows if x["workload"] == name and x["variant"] == variant
                          and x["phase"] == "timed" and (round_index == "all" or x["round"] == round_index)]
                summary.append({
                    "workload": name, "variant": variant, "round": round_index,
                    "compile_ms": describe([x["compile_wall_ns"] / 1e6 for x in subset]),
                    "mask_us_per_token": describe([x["mask_mean_us_per_token"] for x in subset]),
                    "compiled_bytes": describe([x["compiled_bytes"] for x in subset]),
                    "tokens_per_replay": subset[0]["tokens"],
                })
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "verification.json", {
        "passed": True, "records": len(rows),
        "timed_records": sum(x["phase"] == "timed" for x in rows),
        "timed_mask_calls": sum(x["tokens"] for x in rows if x["phase"] == "timed"),
        "mask_equivalence": equivalence,
    })
    lines = [
        "# TagDispatch ablation", "",
        "Explicit TagDispatch is compared with an ordinary right-linear EBNF grammar that encodes the same DFA over valid UTF-8 output. Both recognize text up to the first complete tool tag, followed by that tool's body.", "",
        f"Each grammar runs {args.warmups} warmups and {args.iterations} timed compile-and-replay iterations across {args.rounds} fresh-process rounds. Compilation caching is disabled; one compiler thread and the GPT-2 tokenizer are used.", "",
        "| Tags | Ordinary compile (ms) | TagDispatch compile (ms) | Compile speedup | Ordinary mask (µs/token) | TagDispatch mask (µs/token) | Compiled memory reduction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for count in TAG_COUNTS:
        name = f"tags_{count}"
        ordinary = next(x for x in summary if x["workload"] == name and x["variant"] == "ordinary-ebnf" and x["round"] == "all")
        dispatch = next(x for x in summary if x["workload"] == name and x["variant"] == "tag-dispatch" and x["round"] == "all")
        oc, dc = ordinary["compile_ms"]["median"], dispatch["compile_ms"]["median"]
        om, dm = ordinary["mask_us_per_token"]["median"], dispatch["mask_us_per_token"]["median"]
        ob, db = ordinary["compiled_bytes"]["median"], dispatch["compiled_bytes"]["median"]
        lines.append(f"| {count} | {oc:.3f} | {dc:.3f} | {oc / dc:.2f}× | {om:.3f} | {dm:.3f} | {(1 - db / ob) * 100:.1f}% |")
    lines += [
        "", "Compilation speedups use pooled median wall time. Mask values use pooled medians of replay-average per-token mask latency. Python API and timer overhead are included equally.",
        "", "Allowed-token masks matched at every replayed prefix after excluding GPT-2 tokens whose individual decoded byte piece is not standalone valid UTF-8. TagDispatch intentionally accepts arbitrary raw-byte text, while ordinary character-level EBNF rejects those standalone byte pieces. Fixed valid-UTF-8 completions and EOS behavior matched; the ablation therefore applies to valid-UTF-8 output and is not a proof over arbitrary byte streams.",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    metadata = json.loads((args.output / "metadata.json").read_text())
    metadata.update(status="complete", finished_utc=utc())
    write_json(args.output / "metadata.json", metadata)
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    assert min(args.iterations, args.warmups, args.rounds) > 0
    if args.summarize_only:
        summarize(args)
        return
    if args.worker:
        worker(args)
        return
    prepare(args)
    package = ROOT / "profiling-builds/compressed/package"
    for round_index in range(args.rounds):
        order = VARIANTS if round_index % 2 == 0 else list(reversed(VARIANTS))
        for variant in order:
            env = dict(os.environ, PYTHONPATH=str(package), TOKENIZERS_PARALLELISM="false",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            subprocess.run([
                sys.executable, str(Path(__file__).resolve()), "--worker",
                "--output", str(args.output), "--variant", variant,
                "--round", str(round_index), "--iterations", str(args.iterations),
                "--warmups", str(args.warmups), "--rounds", str(args.rounds),
                "--seed", str(args.seed),
            ], env=env, check=True)
    summarize(args)


if __name__ == "__main__":
    main()
