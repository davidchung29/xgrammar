"""Compare adaptive token masks with a native full-vocabulary simulation baseline."""
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

VARIANTS = ["adaptive-on", "adaptive-off"]


def prepare(args):
    from transformers import AutoTokenizer
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "profiling-results/full/tokenizer", local_files_only=True)
    tokenizer.save_pretrained(out / "tokenizer")
    original = json.loads((ROOT / "profiling-results/full/workloads.json").read_text())
    workloads = []
    for name in ["small", "nested"]:
        old = original[name]
        workloads.append({"name": f"schema_{name}", "kind": "schema", "schema": old["schema_text"],
                          "completion": old["completion"], "token_ids": old["token_ids"]})
    completion = '[1,2,-3,42]'
    array_schema = {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 5}
    workloads.append({"name": "schema_array", "kind": "schema", "schema": json.dumps(array_schema, separators=(",", ":")), "completion": completion,
                      "token_ids": tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.eos_token_id]})
    for w in workloads:
        w["sha256"] = sha(json.dumps(w, sort_keys=True).encode())
    write_json(out / "workloads.json", workloads)
    write_json(out / "metadata.json", {"status": "prepared", "arguments": {**vars(args), "output": str(out)},
               "environment": environment(), "source_commit": cmd("git", "rev-parse", "HEAD"),
               "script_sha256": sha(Path(__file__).read_bytes()),
               "helper_sha256": sha((ROOT / "examples/benchmark/profile_ablations.py").read_bytes()),
               "tokenizer_files": {p.name: sha(p.read_bytes()) for p in (out / "tokenizer").iterdir()},
               "description": "Synthetic ordinary-JSON workloads; GPT-2 vocabulary; no LLM. Runtime-only ablation. Adaptive tables are constructed in both builds, but the off build does not consult them during mask filling."})
    shutil.copyfile(__file__, out / "benchmark_script.py")
    shutil.copyfile(ROOT / "examples/benchmark/profile_ablations.py", out / "profile_ablations.py")
    (out / "source.diff").write_text(cmd("git", "diff", "HEAD")["stdout"])
    for variant in VARIANTS:
        target = out / "builds" / variant
        target.mkdir(parents=True)
        shutil.copyfile(ROOT / f"build-profile-{variant}/CMakeCache.txt", target / "CMakeCache.txt")
        shutil.copyfile(ROOT / f"profiling-builds/{variant}/build.log", target / "build.log")
        wheel, = (ROOT / "profiling-builds" / variant).glob("*.whl")
        write_json(target / "wheel.json", {"path": str(wheel), "sha256": sha(wheel.read_bytes())})


def get_trace(compiled, ids, xgr):
    matcher = xgr.GrammarMatcher(compiled)
    vocab_size = compiled.tokenizer_info.vocab_size
    mask = xgr.allocate_token_bitmask(1, vocab_size)
    hashes = []
    for i, token in enumerate(ids):
        mask.fill_(-1)
        if not matcher.fill_next_token_bitmask(mask):
            mask.fill_(-1)
        if vocab_size % 32:
            mask[0, -1] &= (1 << (vocab_size % 32)) - 1
        hashes.append(sha(mask.numpy().tobytes()))
        assert matcher.accept_token(token), (i, token)
    assert matcher.is_terminated()
    return {"prefix_mask_sha256": hashes, "prefixes": len(ids), "terminated": True}


def worker(args):
    import psutil
    import torch
    import xgrammar as xgr
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    out = args.output / f"{args.variant}-round-{args.round}"
    out.mkdir()
    assert Path(xgr.__file__).is_relative_to(ROOT / f"profiling-builds/{args.variant}/package")
    meta = {"status": "running", "variant": args.variant, "round": args.round,
            "environment": environment(), "package_path": xgr.__file__,
            "binary_hashes": {str(p): sha(p.read_bytes()) for p in Path(xgr.__file__).parent.glob("*.dylib")},
            "timer": vars(time.get_clock_info("perf_counter")),
            "timed_scope": "Per-token mask API calls and acceptance API calls separately. Python/timer overhead included. Compilation, matcher creation, allocation, telemetry and correctness hashing excluded. Off path includes one matcher copy per mask plus native token simulation/rollback; it performs no adaptive-mask lookup."}
    write_json(out / "metadata.json", meta)
    tokenizer = AutoTokenizer.from_pretrained(args.output / "tokenizer", local_files_only=True)
    info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(info, cache_enabled=False, max_threads=1)
    workloads = json.loads((args.output / "workloads.json").read_text())
    random.Random(args.seed + args.round).shuffle(workloads)
    checks = []
    process = psutil.Process()
    try:
        with (out / "iterations.jsonl").open("w", buffering=1) as log:
            for w in workloads:
                start = time.perf_counter_ns()
                compiled = compiler.compile_json_schema(w["schema"]) if w["kind"] == "schema" else compiler.compile_builtin_json_grammar()
                compilation_ns = time.perf_counter_ns() - start
                checks.append({"workload": w["name"], **get_trace(compiled, w["token_ids"], xgr),
                               "compiled_bytes": compiled.memory_size_bytes})
                write_json(out / "correctness.json", checks)
                (out / f"{w['name']}.ebnf").write_text(str(compiled.grammar))
                for phase, count in [("warmup", args.warmups), ("timed", args.iterations)]:
                    for iteration in range(count):
                        before = snap(process)
                        started = utc()
                        timing = replay(compiled, w["token_ids"], xgr)
                        entry = {"variant": args.variant, "round": args.round, "workload": w["name"],
                                 "workload_sha256": w["sha256"], "phase": phase, "iteration": iteration,
                                 "started_utc": started, "before": before, "after": snap(process),
                                 "cache_enabled": False, "adaptive_cache_used": args.variant == "adaptive-on",
                                 "max_threads": 1, "vocab_size": info.vocab_size,
                                 "compiled_bytes": compiled.memory_size_bytes,
                                 "setup_compile_ns_not_timed": compilation_ns, **timing, "status": "ok"}
                        log.write(json.dumps(entry) + "\n")
                print(f"{args.variant} round {args.round+1}: {w['name']} complete ({len(w['token_ids'])} prefixes)", flush=True)
        meta.update(status="complete", finished_environment=environment())
    except BaseException as exc:
        meta.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(out / "metadata.json", meta)


def summarize(args):
    rows, checks = [], []
    for r in range(args.rounds):
        traces = []
        for variant in VARIANTS:
            directory = args.output / f"{variant}-round-{r}"
            assert json.loads((directory / "metadata.json").read_text())["status"] == "complete"
            traces.append(sorted(json.loads((directory / "correctness.json").read_text()), key=lambda c:c["workload"]))
            records = [json.loads(line) for line in (directory / "iterations.jsonl").read_text().splitlines()]
            for w in ["schema_small", "schema_nested", "schema_array"]:
                for phase, n in [("warmup",args.warmups),("timed",args.iterations)]:
                    assert sum(x["workload"]==w and x["phase"]==phase for x in records)==n
            for x in records:
                assert len(x["mask_ns_by_token"])==len(x["accept_ns_by_token"])==x["tokens"]
                assert sum(x["mask_ns_by_token"])==x["mask_total_ns"]
                assert sum(x["accept_ns_by_token"])==x["accept_total_ns"]
            rows.extend(records)
        assert traces[0] == traces[1], f"Mask or compiled-size mismatch in round {r}"
        checks.append({"round": r, "passed":True, "prefixes_compared":sum(x["prefixes"] for x in traces[0])})
    summary=[]
    for w in ["schema_small", "schema_nested", "schema_array"]:
        for variant in VARIANTS:
            for r in [*range(args.rounds), "all"]:
                subset=[x for x in rows if x["workload"]==w and x["variant"]==variant and x["phase"]=="timed" and (r=="all" or x["round"]==r)]
                summary.append({"workload":w,"variant":variant,"round":r,
                                "mask_us_per_token":describe([x["mask_mean_us_per_token"] for x in subset]),
                                "accept_us_per_token":describe([x["accept_mean_us_per_token"] for x in subset]),
                                "replay_ms":describe([x["replay_wall_ns"]/1e6 for x in subset]),"tokens_per_replay":subset[0]["tokens"]})
    write_json(args.output / "summary.json",summary)
    write_json(args.output / "verification.json",{"passed":True,"records":len(rows),"timed_records":sum(x["phase"]=="timed" for x in rows),"timed_mask_calls":sum(x["tokens"] for x in rows if x["phase"]=="timed"),"mask_equivalence":checks})
    lines=["# Adaptive token-mask cache: runtime ablation","",
           "Synthetic ordinary-JSON workloads with the saved GPT-2 vocabulary (50,257 tokens), one compiler thread, and Release builds.","",
           f"Each workload/build runs {args.warmups} warmup replays followed by {args.iterations} timed replays, across {args.rounds} fresh-process rounds. Build order alternates and workload order is shuffled. Each replay starts with a fresh matcher.","",
           "| Workload | Prefixes/replay | Full scan µs/token | Adaptive µs/token | Mask-generation speedup |","|---|---:|---:|---:|---:|"]
    for w in ["schema_small","schema_nested","schema_array"]:
        a=next(x for x in summary if x["workload"]==w and x["variant"]=="adaptive-off" and x["round"]=="all")
        b=next(x for x in summary if x["workload"]==w and x["variant"]=="adaptive-on" and x["round"]=="all")
        ta,tb=a["mask_us_per_token"]["median"],b["mask_us_per_token"]["median"]
        lines.append(f"| {w} | {a['tokens_per_replay']} | {ta:.3f} | {tb:.3f} | {ta/tb:.2f}× |")
    lines += ["", "The full-scan baseline simulates every ordinary vocabulary token in C++, rolls back accepted trials, and handles stop tokens according to parser completion. It copies the matcher once per mask to preserve state. This copy and token-simulation overhead are included in the baseline; there is no Python loop over vocabulary candidates.",
              "", "Both builds retain the same Earley parser, grammar optimizations, repetition compression and tokenizer. Compilation-result caching is disabled. Adaptive tables are still built in both versions; the baseline bypasses them only at runtime. Compilation is excluded, so this is not a measurement of preprocessing savings or end-to-end LLM speedup.",
              "", "Outside timing, both builds produced identical full allowed-token mask hashes at every saved prefix, including EOS positions, with identical compiled-size estimates. Token acceptance and termination also passed. Coverage is limited to these ordinary CFG workloads; the benchmark baseline explicitly rejects budget/capture/token-edge grammars.",
              "", "Speedups are ratios of pooled median replay-average mask latency. Per-round medians, quartiles, all per-token mask/acceptance times, process telemetry, hardware/power conditions, exact inputs/tokenizer files, source patch and binary/build hashes are saved. Times include Python API and timer overhead."]
    lines += ["", "An initial unconstrained-recursive-JSON workload was excluded after mask-equivalence validation failed. At a saved prefix inside a string, cached masking rejected GPT-2 token 42785 (a string/object/array-closing token) although direct parser acceptance permitted a valid JSON continuation. The original failed smoke data and a standalone reproduction are preserved; this table covers only the three workloads that passed equality checks."]
    (args.output / "RESULTS.md").write_text("\n".join(lines)+"\n")
    meta=json.loads((args.output / "metadata.json").read_text());meta.update(status="complete",finished_utc=utc());write_json(args.output / "metadata.json",meta)
    print("\n".join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--iterations",type=int,default=50)
    p.add_argument("--warmups",type=int,default=5)
    p.add_argument("--rounds",type=int,default=3)
    p.add_argument("--seed",type=int,default=20260923)
    p.add_argument("--worker",action="store_true")
    p.add_argument("--variant",choices=VARIANTS)
    p.add_argument("--round",type=int,default=0)
    args=p.parse_args();args.output=args.output.resolve()
    assert min(args.iterations,args.warmups,args.rounds)>0
    if args.worker:
        worker(args);return
    prepare(args)
    for r in range(args.rounds):
        for variant in (VARIANTS if r%2==0 else list(reversed(VARIANTS))):
            env=dict(os.environ,PYTHONPATH=str(ROOT / f"profiling-builds/{variant}/package"),TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
            subprocess.run([sys.executable,str(Path(__file__).resolve()),"--worker","--output",str(args.output),"--variant",variant,"--round",str(r),"--iterations",str(args.iterations),"--warmups",str(args.warmups),"--rounds",str(args.rounds),"--seed",str(args.seed)],env=env,check=True)
    summarize(args)


if __name__=="__main__":
    main()
