"""Reproducible CPU compilation profiling; see profiling.md for interpretation."""
import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

import psutil
import torch
from transformers import AutoTokenizer
import xgrammar as xgr


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True)
    return {"returncode": result.returncode, "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def workloads():
    small = {"type": "object", "properties": {"name": {"type": "string"},
             "age": {"type": "integer"}}, "required": ["name", "age"],
             "additionalProperties": False}
    nested = {"type": "object", "properties": {
        "people": {"type": "array", "items": small},
        "meta": {"type": "object", "properties": {"active": {"type": "boolean"}},
                 "required": ["active"], "additionalProperties": False}},
        "required": ["people", "meta"], "additionalProperties": False}
    props = {f"field_{i:03d}": {"type": "string", "enum": [f"value_{i}_a", f"value_{i}_b"]}
             for i in range(60)}
    large = {"type": "object", "properties": props, "required": list(props),
             "additionalProperties": False}
    return {
        "small": (small, {"name": "Ada", "age": 37}),
        "nested": (nested, {"people": [{"name": "Ada", "age": 37}], "meta": {"active": True}}),
        "large": (large, {key: value["enum"][0] for key, value in props.items()}),
    }


def snapshot(process):
    return {"rss_bytes": process.memory_info().rss,
            "cpu_times_seconds": process.cpu_times()._asdict(),
            "context_switches": process.num_ctx_switches()._asdict(),
            "process_threads": process.num_threads(), "load_average": os.getloadavg(),
            "available_memory_bytes": psutil.virtual_memory().available}


def verify(grammars, ids, vocab_size):
    matchers = [xgr.GrammarMatcher(grammar) for grammar in grammars]
    masks = [xgr.allocate_token_bitmask(1, vocab_size) for _ in matchers]
    trace = hashlib.sha256()
    for position, token in enumerate(ids):
        for matcher, mask in zip(matchers, masks):
            mask.fill_(-1)
            needed = matcher.fill_next_token_bitmask(mask)
            if not needed:
                mask.fill_(-1)
        assert all(torch.equal(masks[0], mask) for mask in masks[1:]), position
        trace.update(masks[0].numpy().tobytes())
        assert all(matcher.accept_token(token) for matcher in matchers), (position, token)
    assert all(matcher.is_terminated() for matcher in matchers)
    return {"passed": True, "prefixes_checked": len(ids), "mask_trace_sha256": trace.hexdigest(),
            "scope": "Mask equivalence along one valid sequence plus EOS per workload; not exhaustive."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", default="openai-community/gpt2")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--hit-batch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    assert min(args.warmups, args.iterations, args.rounds, args.threads, args.hit_batch) > 0
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    process = psutil.Process()
    root = Path(__file__).resolve().parents[2]
    git = lambda *a: command("git", "-C", str(root), *a)
    torch.set_num_threads(1)
    meta = {"started_utc": now(), "arguments": {**vars(args), "output": str(out)},
            "command": sys.argv, "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu_model": command("sysctl", "-n", "machdep.cpu.brand_string"),
            "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
            "ram_bytes": psutil.virtual_memory().total,
            "power_start": command("pmset", "-g", "batt"),
            "thermal_start": command("pmset", "-g", "therm"),
            "git_head": git("rev-parse", "HEAD"), "git_describe": git("describe", "--tags", "--always"),
            "git_status": git("status", "--short"), "submodules": git("submodule", "status"),
            "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()},
            "xgrammar_module": xgr.__file__,
            "xgrammar_install_origin": metadata.distribution("xgrammar").read_text("direct_url.json"),
            "script_sha256": digest(Path(__file__).read_bytes()),
            "timer": vars(time.get_clock_info("perf_counter")),
            "environment": {k: os.environ.get(k) for k in
                            ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM"]},
            "timed_scope": "compile_json_schema only, including Python call/loop; setup, telemetry, validation, and result destruction excluded",
            "status": "running"}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    (out / "source.diff").write_text(git("diff", "HEAD")["stdout"])
    (out / "benchmark_script.py").write_bytes(Path(__file__).read_bytes())
    build_cache = root / "build" / "CMakeCache.txt"
    if build_cache.exists():
        (out / "CMakeCache.txt").write_bytes(build_cache.read_bytes())
    meta["native_libraries_sha256"] = {str(p): digest(p.read_bytes()) for p in
        Path(xgr.__file__).parent.rglob("*") if p.suffix in {".so", ".dylib"}}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    tokenizer.save_pretrained(out / "tokenizer")
    info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    meta["tokenizer"] = {"name": args.tokenizer, "revision_requested": args.revision,
                         "resolved_commit": tokenizer.init_kwargs.get("_commit_hash"),
                         "vocab_size": info.vocab_size,
                         "files_sha256": {p.name: digest(p.read_bytes()) for p in
                                           (out / "tokenizer").iterdir() if p.is_file()}}
    data = {}
    for name, (schema, completion) in workloads().items():
        schema_text = json.dumps(schema, separators=(",", ":"))
        text = json.dumps(completion, separators=(",", ":"))
        ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
        data[name] = {"schema": schema, "schema_text": schema_text,
                      "schema_sha256": digest(schema_text.encode()), "completion": text,
                      "token_ids": ids}
    (out / "workloads.json").write_text(json.dumps(data, indent=2))
    configs = {"uncached_1t": (False, 1), "uncached_nt": (False, args.threads),
               "warm_cache_1t": (True, 1), "cold_cache_1t": (True, 1)}
    meta["configurations"] = configs
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    rng = random.Random(args.seed)
    records, checks = [], []
    with (out / "iterations.jsonl").open("w", buffering=1) as log:
        for round_id in range(args.rounds):
            names = list(data)
            rng.shuffle(names)
            for name in names:
                workload = data[name]
                schema = workload["schema_text"]
                compilers = {key: xgr.GrammarCompiler(info, cache_enabled=cache, max_threads=threads)
                             for key, (cache, threads) in configs.items()}
                for phase, count in [("warmup", args.warmups), ("timed", args.iterations)]:
                    for iteration in range(count):
                        order = list(configs)
                        rng.shuffle(order)
                        for order_index, config in enumerate(order):
                            if config == "cold_cache_1t":
                                compilers[config] = xgr.GrammarCompiler(info, cache_enabled=True, max_threads=1)
                            compiler = compilers[config]
                            # First warm-cache warmup is a separately labelled cache miss.
                            prime = config == "warm_cache_1t" and phase == "warmup" and iteration == 0
                            calls = args.hit_batch if config == "warm_cache_1t" and not prime else 1
                            results = [None] * calls
                            before = snapshot(process)
                            started = now()
                            cpu_start = time.process_time_ns()
                            start = time.perf_counter_ns()
                            for j in range(calls):
                                results[j] = compiler.compile_json_schema(schema)
                            elapsed = time.perf_counter_ns() - start
                            cpu_elapsed = time.process_time_ns() - cpu_start
                            after = snapshot(process)
                            record = {"sequence": len(records), "started_utc": started,
                                      "round": round_id, "workload": name,
                                      "schema_sha256": workload["schema_sha256"], "phase": phase,
                                      "iteration": iteration, "order_index": order_index,
                                      "configuration": config, "cache_enabled": configs[config][0],
                                      "max_threads": configs[config][1], "cache_prime": prime,
                                      "calls": calls, "elapsed_ns": elapsed, "ns_per_call": elapsed / calls,
                                      "process_cpu_ns": cpu_elapsed,
                                      "compiled_memory_bytes": results[-1].memory_size_bytes,
                                      "before": before, "after": after, "status": "ok"}
                            log.write(json.dumps(record) + "\n")
                            records.append(record)
                            del results
                grammars = [compilers[key].compile_json_schema(schema) for key in configs]
                checks.append({"round": round_id, "workload": name,
                               **verify(grammars, workload["token_ids"], info.vocab_size)})
                (out / "correctness.json").write_text(json.dumps(checks, indent=2))
                print(f"Round {round_id + 1}/{args.rounds}: {name} complete; mask checks passed", flush=True)
    summary = []
    for name in data:
        for config in configs:
            for round_id in [*range(args.rounds), "all"]:
                values = [r["ns_per_call"] / 1e6 for r in records
                          if r["phase"] == "timed" and r["workload"] == name
                          and r["configuration"] == config
                          and (round_id == "all" or r["round"] == round_id)]
                quartiles = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else values * 3
                summary.append({"workload": name, "configuration": config, "round": round_id,
                                "n": len(values), "median_ms": statistics.median(values),
                                "p25_ms": quartiles[0], "p75_ms": quartiles[2],
                                "min_ms": min(values), "max_ms": max(values)})
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# CPU compilation profiling results", "", "All times are milliseconds per compile API call.", "",
             "| Workload | Configuration | Median | P25–P75 | Speedup vs uncached 1 thread |",
             "|---|---|---:|---:|---:|"]
    for name in data:
        rows = [s for s in summary if s["round"] == "all" and s["workload"] == name]
        baseline = next(s["median_ms"] for s in rows if s["configuration"] == "uncached_1t")
        for s in rows:
            lines.append(f"| {name} | {s['configuration']} | {s['median_ms']:.6f} | {s['p25_ms']:.6f}–{s['p75_ms']:.6f} | {baseline / s['median_ms']:.2f}× |")
    lines += ["", "Warm-cache results are batches of repeated identical-schema cache hits, not decoding speedups.",
              "Cold-cache results start with an empty compiler cache; compiler construction is excluded.",
              "Correctness checks compare all configurations along the saved valid sequences, including EOS; they are not exhaustive.",
              "RSS is process-wide and affected by allocator reuse; it is not per-compile peak memory.",
              "Rounds share one process. CPU load, scheduling, and thermal variation are not controlled.",
              "See summary.json for per-round statistics and iterations.jsonl for every sample."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    meta.update(status="complete", finished_utc=now(), records=len(records),
                power_end=command("pmset", "-g", "batt"), thermal_end=command("pmset", "-g", "therm"))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
