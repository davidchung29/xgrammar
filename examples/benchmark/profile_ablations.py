"""CPU optimization ablations. Run from the profiling checkout; see profiling.md."""
import argparse
import collections
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ["disabled", "fresh_per_request", "shared_across_requests"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def cmd(*args):
    result = subprocess.run(args, capture_output=True, text=True)
    return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def environment():
    import psutil
    return {"utc": utc(), "platform": platform.platform(), "architecture": platform.machine(),
            "python": sys.version, "executable": sys.executable,
            "cpu": cmd("sysctl", "-n", "machdep.cpu.brand_string"),
            "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total,
            "power": cmd("pmset", "-g", "batt"), "power_settings": cmd("pmset", "-g", "custom"),
            "thermal": cmd("pmset", "-g", "therm"),
            "dependencies": {d.metadata["Name"]: d.version for d in md.distributions()},
            "thread_environment": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM"]}}


def snap(process):
    import psutil
    return {"rss_bytes": process.memory_info().rss, "cpu_seconds": process.cpu_times()._asdict(),
            "context_switches": process.num_ctx_switches()._asdict(),
            "threads": process.num_threads(), "load_average": os.getloadavg(),
            "available_ram_bytes": psutil.virtual_memory().available}


def prepare(args):
    from transformers import AutoTokenizer
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer.save_pretrained(out / "tokenizer")
    def ids(text):
        return tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
    pool = []
    for i in range(40):
        arguments = {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"},
            "category": {"type": "string", "enum": [f"category_{i}_a", f"category_{i}_b"]}},
            "required": ["query", "limit", "category"], "additionalProperties": False}
        pool.append({"type": "object", "properties": {
            "name": {"type": "string", "const": f"tool_{i:02d}"}, "arguments": arguments},
            "required": ["name", "arguments"], "additionalProperties": False})
    rng = random.Random(args.seed)
    streams = []
    for size in [4, 8, 16]:
        seen, requests = set(), []
        while len(requests) < args.requests:
            selected = tuple(sorted(rng.sample(range(len(pool)), size)))
            if selected in seen:
                continue
            seen.add(selected)
            schema = json.dumps({"anyOf": [pool[i] for i in selected]}, separators=(",", ":"))
            i = selected[0]
            text = json.dumps({"name": f"tool_{i:02d}", "arguments": {
                "query": "example", "limit": 3, "category": f"category_{i}_a"}}, separators=(",", ":"))
            requests.append({"request_index": len(requests), "tool_ids": selected, "schema": schema,
                             "sha256": sha(schema.encode()), "completion": text, "token_ids": ids(text)})
        assert len({r["sha256"] for r in requests}) == len(requests)
        streams.append({"name": f"tools_{size}", "tools_per_request": size, "requests": requests})
    repeats = []
    def body(n):
        return "".join(["a1;", "b2;", "z9;"][i % 3] for i in range(n)) + "\n"
    for lower in [100, 500, 1000]:
        upper = lower + 8
        grammar = f'root ::= item{{{lower},{upper}}} "\\n"\nitem ::= [a-z] [0-9] ";"\n'
        repeats.append({"name": f"repeat_{lower}_{upper}", "lower": lower, "upper": upper,
                        "grammar": grammar, "sha256": sha(grammar.encode()),
                        "completion": body(lower + 4), "token_ids": ids(body(lower + 4)),
                        "validations": [{"count": n, "text": body(n),
                                         "expected_valid": lower <= n <= upper,
                                         "token_ids": ids(body(n)) if lower <= n <= upper else None}
                                        for n in [lower - 1, lower, upper, upper + 1]]})
    write_json(out / "workloads.json", {"tool_pool": pool, "cross_grammar": streams, "repetition": repeats})
    write_json(out / "metadata.json", {"status": "prepared", "arguments": {**vars(args), "output": str(out)},
               "environment": environment(), "git_head": cmd("git", "rev-parse", "HEAD"),
               "git_status": cmd("git", "status", "--short"), "script_sha256": sha(Path(__file__).read_bytes()),
               "tokenizer_files": {p.name: sha(p.read_bytes()) for p in (out / "tokenizer").iterdir()}})
    (out / "source.diff").write_text(cmd("git", "diff", "HEAD")["stdout"])
    shutil.copyfile(__file__, out / "benchmark_script.py")
    for variant in ["compressed", "expanded"]:
        target = out / "builds" / variant
        target.mkdir(parents=True)
        shutil.copyfile(ROOT / f"build-profile-{variant}" / "CMakeCache.txt", target / "CMakeCache.txt")
        shutil.copyfile(ROOT / "profiling-builds" / variant / "build.log", target / "build.log")
        wheels = list((ROOT / "profiling-builds" / variant).glob("*.whl"))
        assert len(wheels) == 1
        write_json(target / "wheel.json", {"path": str(wheels[0]), "sha256": sha(wheels[0].read_bytes())})


def trace(compiled, token_ids, xgr):
    matcher = xgr.GrammarMatcher(compiled)
    mask = xgr.allocate_token_bitmask(1, compiled.tokenizer_info.vocab_size)
    digest = hashlib.sha256()
    for i, token in enumerate(token_ids):
        mask.fill_(-1)
        if not matcher.fill_next_token_bitmask(mask):
            mask.fill_(-1)
        remainder = compiled.tokenizer_info.vocab_size % 32
        if remainder:
            mask[0, -1] &= (1 << remainder) - 1
        digest.update(mask.numpy().tobytes())
        digest.update(token.to_bytes(4, "little"))
        assert matcher.accept_token(token), (i, token)
    assert matcher.is_terminated()
    return {"mask_trace_sha256": digest.hexdigest(), "prefixes": len(token_ids), "terminated": True}


def replay(compiled, token_ids, xgr):
    matcher = xgr.GrammarMatcher(compiled)
    mask = xgr.allocate_token_bitmask(1, compiled.tokenizer_info.vocab_size)
    mask_times, accept_times = [0] * len(token_ids), [0] * len(token_ids)
    cpu_start = time.process_time_ns()
    replay_start = time.perf_counter_ns()
    for i, token in enumerate(token_ids):
        start = time.perf_counter_ns()
        matcher.fill_next_token_bitmask(mask)
        mask_times[i] = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        accepted = matcher.accept_token(token)
        accept_times[i] = time.perf_counter_ns() - start
        assert accepted, (i, token)
    elapsed = time.perf_counter_ns() - replay_start
    cpu = time.process_time_ns() - cpu_start
    assert matcher.is_terminated()
    return {"replay_wall_ns": elapsed, "replay_cpu_ns": cpu, "tokens": len(token_ids),
            "mask_ns_by_token": mask_times, "accept_ns_by_token": accept_times,
            "mask_total_ns": sum(mask_times), "accept_total_ns": sum(accept_times),
            "mask_mean_us_per_token": sum(mask_times) / len(token_ids) / 1000,
            "accept_mean_us_per_token": sum(accept_times) / len(token_ids) / 1000}


def compile_sample(compiler, workload, kind, base, log, process, xgr):
    before = snap(process)
    cache_before = compiler.get_cache_size_bytes()
    if base["configuration"] != "shared_across_requests" or base.get("request_index") == 0:
        assert cache_before == 0, (base, cache_before)
    started = utc()
    cpu_start = time.process_time_ns()
    start = time.perf_counter_ns()
    if kind == "cross":
        compiled = compiler.compile_json_schema(workload["schema"])
    else:
        compiled = compiler.compile_grammar(workload["grammar"])
    elapsed = time.perf_counter_ns() - start
    cpu_elapsed = time.process_time_ns() - cpu_start
    after_compile = snap(process)
    entry = {**base, "started_utc": started, "workload_sha256": workload["sha256"],
             "compile_wall_ns": elapsed, "compile_cpu_ns": cpu_elapsed,
             "cache_before_bytes": cache_before, "cache_after_bytes": compiler.get_cache_size_bytes(),
             "compiled_memory_bytes": compiled.memory_size_bytes, "before": before,
             "after_compile": after_compile, "max_threads": 1,
             **replay(compiled, workload["token_ids"], xgr), "after_replay": snap(process), "status": "ok"}
    log.write(json.dumps(entry) + "\n")
    return entry


def worker(args):
    import psutil
    import torch
    import xgrammar as xgr
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    expected = ROOT / "profiling-builds" / args.variant / "package"
    assert Path(xgr.__file__).is_relative_to(expected), xgr.__file__
    output = args.output / f"{args.worker}-{args.variant}-round-{args.round}"
    output.mkdir()
    meta = {"status": "running", "environment": environment(), "worker": args.worker,
            "variant": args.variant, "round": args.round, "xgrammar_path": xgr.__file__,
            "script_sha256": sha(Path(__file__).read_bytes()),
            "binary_hashes": {str(p): sha(p.read_bytes()) for p in Path(xgr.__file__).parent.glob("*.dylib")},
            "timer": vars(time.get_clock_info("perf_counter")),
            "measurement_scope": "Compilation includes grammar parsing/optimization and token-mask precomputation. Token replay times mask filling and acceptance separately using per-call wall timers; timer overhead is included. Setup, telemetry, mask hashing and result destruction are outside compile timing."}
    write_json(output / "metadata.json", meta)
    process = psutil.Process()
    tokenizer = AutoTokenizer.from_pretrained(args.output / "tokenizer", local_files_only=True)
    info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    workloads = json.loads((args.output / "workloads.json").read_text())
    checks = []
    rng = random.Random(args.seed + args.round)
    try:
        with (output / "iterations.jsonl").open("w", buffering=1) as log:
            if args.worker == "cross":
                streams = list(workloads["cross_grammar"])
                rng.shuffle(streams)
                for stream in streams:
                    traces = {}
                    for config in CONFIGS:
                        compiler = xgr.GrammarCompiler(info, cache_enabled=config != "disabled", max_threads=1)
                        traces[config] = []
                        for request in stream["requests"]:
                            if config == "fresh_per_request":
                                compiler = xgr.GrammarCompiler(info, cache_enabled=True, max_threads=1)
                            compiled = compiler.compile_json_schema(request["schema"])
                            traces[config].append(trace(compiled, request["token_ids"], xgr))
                    assert traces[CONFIGS[0]] == traces[CONFIGS[1]] == traces[CONFIGS[2]]
                    checks.append({"workload": stream["name"], "passed": True, "traces": traces})
                    write_json(output / "correctness.json", checks)
                    for phase, count in [("warmup", args.warmups), ("timed", args.iterations)]:
                        for iteration in range(count):
                            order = list(CONFIGS)
                            rng.shuffle(order)
                            for order_index, config in enumerate(order):
                                compiler = xgr.GrammarCompiler(info, cache_enabled=config != "disabled", max_threads=1)
                                assert compiler.get_cache_size_bytes() == 0
                                seen = set()
                                for request in stream["requests"]:
                                    assert request["sha256"] not in seen
                                    seen.add(request["sha256"])
                                    if config == "fresh_per_request":
                                        compiler = xgr.GrammarCompiler(info, cache_enabled=True, max_threads=1)
                                    compile_sample(compiler, request, "cross", {
                                        "experiment": "cross", "variant": args.variant, "round": args.round,
                                        "workload": stream["name"], "phase": phase, "iteration": iteration,
                                        "order_index": order_index, "configuration": config,
                                        "cache_enabled": config != "disabled", "request_index": request["request_index"],
                                        "whole_grammar_hit_possible": False, "tool_ids": request["tool_ids"]}, log, process, xgr)
                    print(f"Cross-grammar round {args.round + 1}: {stream['name']} done", flush=True)
            else:
                repeats = list(workloads["repetition"])
                rng.shuffle(repeats)
                for workload in repeats:
                    compiler = xgr.GrammarCompiler(info, cache_enabled=False, max_threads=1)
                    compiled = compiler.compile_grammar(workload["grammar"])
                    cases = []
                    for case in workload["validations"]:
                        matcher = xgr.GrammarMatcher(compiled)
                        accepted = matcher.accept_string(case["text"])
                        valid = accepted and matcher.is_completed()
                        assert valid == case["expected_valid"], (args.variant, workload["name"], case["count"])
                        cases.append({"count": case["count"], "valid": valid,
                                      "trace": trace(compiled, case["token_ids"], xgr) if valid else None})
                    matcher = xgr.GrammarMatcher(compiled)
                    assert not matcher.accept_string("@")
                    checks.append({"workload": workload["name"], "passed": True, "cases": cases,
                                   "invalid_body_rejected": True})
                    write_json(output / "correctness.json", checks)
                    # Store normalized grammar so the ablation can be inspected directly.
                    (output / f"{workload['name']}.ebnf").write_text(str(compiled.grammar))
                    for phase, count in [("warmup", args.warmups), ("timed", args.iterations)]:
                        for iteration in range(count):
                            compile_sample(compiler, workload, "repeat", {
                                "experiment": "repeat", "variant": args.variant, "round": args.round,
                                "workload": workload["name"], "phase": phase, "iteration": iteration,
                                "configuration": args.variant, "cache_enabled": False,
                                "lower": workload["lower"], "upper": workload["upper"]}, log, process, xgr)
                    print(f"Repetition {args.variant} round {args.round + 1}: {workload['name']} done", flush=True)
        meta.update(status="complete", finished=environment())
    except BaseException as exc:
        meta.update(status="failed", error=repr(exc), finished=environment())
        raise
    finally:
        write_json(output / "metadata.json", meta)


def describe(values):
    if len(values) == 1:
        q = values * 3
    else:
        q = statistics.quantiles(values, n=4, method="inclusive")
    return {"n": len(values), "median": statistics.median(values), "p25": q[0], "p75": q[2],
            "mean": statistics.mean(values), "min": min(values), "max": max(values)}


def summarize(args):
    out = args.output
    all_rows = []
    for directory in sorted(out.glob("*-round-*")):
        assert json.loads((directory / "metadata.json").read_text())["status"] == "complete"
        rows = [json.loads(line) for line in (directory / "iterations.jsonl").read_text().splitlines()]
        all_rows.extend(rows)
    checks = []
    for round_id in range(args.rounds):
        a = json.loads((out / f"repeat-compressed-round-{round_id}" / "correctness.json").read_text())
        b = json.loads((out / f"repeat-expanded-round-{round_id}" / "correctness.json").read_text())
        assert sorted(a, key=lambda r: r["workload"]) == sorted(b, key=lambda r: r["workload"])
        checks.append({"round": round_id, "repetition_mask_traces_and_boundary_checks_equal": True})
    groups = collections.defaultdict(list)
    for row in all_rows:
        if row["phase"] == "timed":
            groups[(row["experiment"], row["workload"], row["configuration"], row["round"])].append(row)
    summary = []
    for experiment, workload, config in sorted({k[:3] for k in groups}):
        for round_id in [*range(args.rounds), "all"]:
            rows = [r for k, v in groups.items() if k[:3] == (experiment, workload, config)
                    and (round_id == "all" or k[3] == round_id) for r in v]
            if experiment == "cross":
                trials = collections.defaultdict(list)
                for r in rows:
                    trials[(r["round"], r["iteration"])].append(r)
                compile_values = [sum(r["compile_wall_ns"] for r in trial) / len(trial) / 1e6 for trial in trials.values()]
                cache_values = [max(trial, key=lambda r: r["request_index"])["cache_after_bytes"] for trial in trials.values()]
            else:
                compile_values = [r["compile_wall_ns"] / 1e6 for r in rows]
                cache_values = [r["cache_after_bytes"] for r in rows]
            summary.append({"experiment": experiment, "workload": workload, "configuration": config,
                            "round": round_id, "compile_ms": describe(compile_values),
                            "mask_us_per_token": describe([r["mask_mean_us_per_token"] for r in rows]),
                            "compiled_bytes": describe([r["compiled_memory_bytes"] for r in rows]),
                            "cache_end_bytes": describe(cache_values)})
    expected_cross = args.rounds * 3 * len(CONFIGS) * args.requests * (args.warmups + args.iterations)
    expected_repeat = args.rounds * 3 * 2 * (args.warmups + args.iterations)
    assert len(all_rows) == expected_cross + expected_repeat, (len(all_rows), expected_cross, expected_repeat)
    count_groups = collections.Counter((r["experiment"], r["round"], r["workload"], r["configuration"], r["phase"]) for r in all_rows)
    for key, value in count_groups.items():
        expected = (args.requests if key[0] == "cross" else 1) * (args.warmups if key[-1] == "warmup" else args.iterations)
        assert value == expected, (key, value, expected)
    write_json(out / "summary.json", summary)
    write_json(out / "verification.json", {"passed": True, "records": len(all_rows),
               "timed_records": sum(r["phase"] == "timed" for r in all_rows), "cross_records": expected_cross,
               "repetition_records": expected_repeat, "per_group_counts_verified": True,
               "cross_build_checks": checks})
    lines = ["# XGrammar optimization profiling ablations", "", "These are synthetic workload ablations on the pinned checkout, not reproductions of the papers' datasets or absolute numbers.", "",
             f"{args.warmups} warmups then {args.iterations} measured repetitions in each of {args.rounds} rounds; one compiler thread; saved GPT-2 vocabulary.", "",
             "## Cross-grammar compilation", "", "Each repetition is a complete stream of distinct grammars. Caches start empty, then are either disabled, recreated before every request, or preserved across requests. Whole-grammar hits are excluded by unique request keys. The primary cross-request comparison is fresh-per-request versus shared.", "",
             "Compilation values are medians of stream-average milliseconds/request. Each stream includes its cold first request. Cache memory includes retained complete grammars as well as rule-level entries.", "",
             "| Tools/request | Disabled ms | Fresh ms | Shared ms | Fresh/shared speedup | Shared end-cache MiB |", "|---|---:|---:|---:|---:|---:|"]
    lookup = {(s["experiment"], s["workload"], s["configuration"]): s for s in summary if s["round"] == "all"}
    for n in [4, 8, 16]:
        rs = [lookup[("cross", f"tools_{n}", c)] for c in CONFIGS]
        a, b, c = [r["compile_ms"]["median"] for r in rs]
        lines.append(f"| {n} | {a:.3f} | {b:.3f} | {c:.3f} | {b/c:.2f}× | {rs[2]['cache_end_bytes']['median']/2**20:.3f} |")
    lines += ["", "## Repetition compression", "", "Both builds use the same source, Release flags and dependencies. The expanded build redirects repetition handling to the existing explicit expansion; memoization and other optimizations remain enabled. Compilation caching is disabled in both builds.", "",
              "| Repetition range | Expanded compile ms | Compressed compile ms | Compile speedup | Expanded mask µs/token | Compressed mask µs/token | Expanded/compressed size KiB |", "|---|---:|---:|---:|---:|---:|---:|"]
    for lower in [100, 500, 1000]:
        name = f"repeat_{lower}_{lower+8}"
        a, b = [lookup[("repeat", name, c)] for c in ["expanded", "compressed"]]
        ca, cb = a["compile_ms"]["median"], b["compile_ms"]["median"]
        lines.append(f"| {lower}–{lower+8} | {ca:.3f} | {cb:.3f} | {ca/cb:.2f}× | {a['mask_us_per_token']['median']:.3f} | {b['mask_us_per_token']['median']:.3f} | {a['compiled_bytes']['median']/1024:.1f} / {b['compiled_bytes']['median']/1024:.1f} |")
    lines += ["", "The 100–108 range is below the default compression threshold (128), so it is a control expected to show little structural difference.", "",
              "## Validation and limitations", "", "All saved cross-configuration mask-trace comparisons passed. Repetition builds agree on masks along lower/upper valid boundary sequences, reject lower−1 and upper+1 completed strings, and reject an invalid body character. These checks are not an exhaustive proof.",
              "", "Per-token timings include Python and timer overhead; compare variants under the same measurement method. Replay excludes an LLM and GPU mask application. Compiled size and cache size are library estimates; process RSS snapshots are not allocation peaks.",
              "", "Configuration order is shuffled for cache streams. Repetition build order alternates by round and workloads are shuffled. Each worker starts a fresh process. See per-round quartiles in summary.json and all raw compile/replay/token measurements in the worker JSONL files.",
              "", "Power and thermal settings are recorded per worker in metadata.json. Hardware, source patch, build logs, CMake caches, binary/wheel hashes, tokenizer and exact inputs are preserved."]
    if args.reuse_repetition_from:
        lines += ["", "The repetition measurements were retained from the earlier validated run; only cache workers were rerun with fresh per-request compilers. Original worker environments/timestamps, binary hashes and script are preserved, with reuse provenance recorded in worker metadata."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    meta = json.loads((out / "metadata.json").read_text())
    meta.update(status="complete", finished_utc=utc())
    write_json(out / "metadata.json", meta)
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", default=str(ROOT / "profiling-results/full/tokenizer"))
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--reuse-repetition-from", help="Reuse validated repetition workers with identical inputs/settings/binaries; rerun only cache workers")
    parser.add_argument("--worker", choices=["cross", "repeat"])
    parser.add_argument("--variant", choices=["compressed", "expanded"])
    parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    assert min(args.iterations, args.warmups, args.rounds, args.requests) > 0
    if args.worker:
        worker(args)
        return
    prepare(args)
    if args.reuse_repetition_from:
        prior = Path(args.reuse_repetition_from).resolve()
        prior_meta = json.loads((prior / "metadata.json").read_text())
        assert prior_meta["status"] in {"complete", "superseded_cache_accounting"}
        for setting in ["warmups", "iterations", "rounds"]:
            assert prior_meta["arguments"][setting] == getattr(args, setting)
        old_inputs = json.loads((prior / "workloads.json").read_text())["repetition"]
        new_inputs = json.loads((args.output / "workloads.json").read_text())["repetition"]
        assert old_inputs == new_inputs
        shutil.copyfile(prior / "benchmark_script.py", args.output / "reused_repetition_script.py")
        for round_id in range(args.rounds):
            for variant in ["compressed", "expanded"]:
                name = f"repeat-{variant}-round-{round_id}"
                source = prior / name
                worker_meta = json.loads((source / "metadata.json").read_text())
                assert worker_meta["status"] == "complete"
                for library, expected_hash in worker_meta["binary_hashes"].items():
                    assert sha(Path(library).read_bytes()) == expected_hash
                shutil.copytree(source, args.output / name)
                worker_meta.update(reused_from=str(source), reused_utc=utc(),
                                   script_sha256=prior_meta["script_sha256"])
                write_json(args.output / name / "metadata.json", worker_meta)
    for round_id in range(args.rounds):
        variants = ["compressed", "expanded"] if round_id % 2 == 0 else ["expanded", "compressed"]
        tasks = [("cross", "compressed")]
        if not args.reuse_repetition_from:
            tasks.extend(("repeat", v) for v in variants)
        for task, variant in tasks:
            env = dict(os.environ, PYTHONPATH=str(ROOT / "profiling-builds" / variant / "package"),
                       TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(args.output),
                            "--worker", task, "--variant", variant, "--round", str(round_id),
                            "--iterations", str(args.iterations), "--warmups", str(args.warmups),
                            "--rounds", str(args.rounds), "--requests", str(args.requests),
                            "--seed", str(args.seed)], env=env, check=True)
    summarize(args)


if __name__ == "__main__":
    main()
