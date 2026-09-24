# XGrammar optimization profiling

These CPU benchmarks measure compilation caching, repetition compression, adaptive
token-mask caching, and TagDispatch on controlled synthetic workloads. They use the GPT-2 tokenizer and
do not run an LLM or use a GPU. The ablations follow the evaluation style of the XGrammar
papers, but they do not reproduce the papers' datasets, vocabularies, hardware, or
end-to-end serving experiments.

All full experiments use one compiler thread unless a configuration explicitly tests
parallel compilation. Each workload runs five warmups followed by fifty timed repetitions
in each of three rounds. Use a new output directory for every run.

## Setup

Initialize the repository and create a native Python environment:

```sh
git submodule update --init --recursive
python3 -m venv .venv-native
.venv-native/bin/python -m pip install -U pip
.venv-native/bin/python -m pip install . psutil transformers torch
```

Build the repository in Release mode. Platform-specific SDK variables may be required;
the benchmark records the final build configuration and native-library hashes.

## General compilation profile

`profile_compilation.py` compares disabled compilation caching, an empty cache, a warm
whole-grammar cache, and one-thread versus multi-thread compilation:

```sh
.venv-native/bin/python examples/benchmark/profile_compilation.py \
  --output profiling-results/compilation \
  --warmups 5 --iterations 50 --rounds 3
```

The first run can download the public GPT-2 tokenizer, but no model weights. Pass
`--tokenizer PATH` to reuse a saved tokenizer. This benchmark measures compilation;
warm whole-grammar cache hits must not be interpreted as decoding or LLM throughput.

## Cross-grammar cache and repetition compression

`profile_ablations.py` runs two optimization ablations:

```sh
.venv-native/bin/python examples/benchmark/profile_ablations.py \
  --output profiling-results/ablations \
  --tokenizer PATH/TO/SAVED/GPT2_TOKENIZER \
  --warmups 5 --iterations 50 --rounds 3 --requests 12
```

### Cross-grammar cache

The benchmark generates a pool of forty synthetic tools. Each request grammar contains
4, 8, or 16 tools, and every stream contains twelve distinct request grammars. The three
configurations are:

1. Compilation caching disabled.
2. Caching enabled with a fresh compiler for every request.
3. Caching enabled with one compiler retained across the request stream.

Every stream starts with an empty cache, complete grammar keys do not repeat, and all
configurations receive the same requests in the same order. The primary comparison is
fresh-per-request versus shared-across-requests, which preserves within-request reuse in
both configurations and isolates reuse between different grammars.

Each compiled request replays a fixed valid completion. Compilation wall and CPU time are
recorded, along with compiled size, cache size, and mask and acceptance latency for every
token. Compilation time is the primary metric; per-token masking checks for runtime
regressions.

### Repetition compression

The repetition workloads accept `[a-z][0-9];` repeated 100–108, 500–508, or 1000–1008
times, followed by a newline. The first range is below the default compression threshold
of 128 and serves as a control.

The normal build is compared with a benchmark-only build that redirects repetition
handling to XGrammar's existing explicit expansion:

```sh
.venv-native/bin/python -m pip wheel . --no-deps \
  --wheel-dir profiling-builds/compressed \
  -Cbuild-dir=build-profile-compressed \
  -Ccmake.define.XGRAMMAR_PROFILE_DISABLE_REPETITION_COMPRESSION=OFF

.venv-native/bin/python -m pip wheel . --no-deps \
  --wheel-dir profiling-builds/expanded \
  -Cbuild-dir=build-profile-expanded \
  -Ccmake.define.XGRAMMAR_PROFILE_DISABLE_REPETITION_COMPRESSION=ON
```

Install each wheel into the corresponding isolated package directory expected by the
harness. Both builds disable compilation-result caching and retain adaptive masks,
memoization, and every other grammar optimization. The benchmark records compilation
time, compiled size, and per-token mask latency.

## Adaptive token-mask cache

`profile_adaptive_mask.py` compares normal adaptive mask generation with a benchmark-only
native baseline that simulates every ordinary vocabulary token, accepts it speculatively,
and rolls back successful trials:

```sh
.venv-native/bin/python -m pip wheel . --no-deps \
  --wheel-dir profiling-builds/adaptive-on \
  -Cbuild-dir=build-profile-adaptive-on \
  -Ccmake.define.XGRAMMAR_PROFILE_FULL_VOCAB_MASK=OFF

.venv-native/bin/python -m pip wheel . --no-deps \
  --wheel-dir profiling-builds/adaptive-off \
  -Cbuild-dir=build-profile-adaptive-off \
  -Ccmake.define.XGRAMMAR_PROFILE_FULL_VOCAB_MASK=ON

.venv-native/bin/python examples/benchmark/profile_adaptive_mask.py \
  --output profiling-results/adaptive \
  --warmups 5 --iterations 50 --rounds 3
```

The three workloads are a small JSON object, a nested object and array schema, and a
bounded integer array. Compilation occurs outside the timed region. Each replay starts
with a fresh matcher and records every mask-generation and token-acceptance call
separately. Compilation caching is disabled in both builds; repetition compression and
other grammar optimizations remain enabled.

The full-vocabulary path supports ordinary CFG workloads only. It intentionally rejects
budget, capture, and token-edge grammars. It is a naive native full-scan baseline, not an
end-to-end LLM comparison or a claim about every possible cache-free implementation.

## TagDispatch

`profile_tag_dispatch.py` compares explicit `TagDispatch` with a generated right-linear
ordinary EBNF grammar that encodes the same tag-prefix DFA:

```sh
.venv-native/bin/python examples/benchmark/profile_tag_dispatch.py \
  --output profiling-results/tag-dispatch \
  --warmups 5 --iterations 50 --rounds 3
```

The workloads contain 10, 50, or 100 tags named `<function=tool_NNN>`, followed by a
small tool-specific body. Each fixed completion contains ordinary text, one middle-index
tag, its valid body, and EOS. Both configurations use the same default Release build,
one compiler thread, disabled compilation caching, and all other optimizations enabled.
Every iteration uses a fresh compiler, then records compilation time, compiled size,
and mask and acceptance latency at every replayed token.

The generated ordinary grammar explicitly represents every proper tag-prefix state and
transition; `TagDispatch` represents the trigger set with its specialized automaton. This
is a synthetic representation ablation rather than a reproduction of the paper's tool
dataset or reported absolute timings.

With GPT-2, 112 vocabulary entries decode to byte fragments that are not independently
valid UTF-8. `TagDispatch` accepts arbitrary raw-byte text, while character-level EBNF
rejects those standalone pieces. The harness therefore requires complete mask equality
for all independently valid UTF-8 pieces and EOS, records the full-mask differences, and
limits equivalence claims to valid UTF-8 output.

## Validation and saved records

Outside timed regions, the harnesses compare complete allowed-token mask hashes at every
saved prefix and verify token acceptance and termination. Repetition validation also
checks the lower and upper bounds, lower-minus-one and upper-plus-one completed strings,
and an invalid body character. These checks are targeted regression tests rather than an
exhaustive correctness proof.

Generated output includes exact workloads and token IDs, per-iteration JSONL records,
per-token mask and acceptance timings, correctness files, summaries, source and binary
hashes, environment information, and readable result tables. Generated `profiling-results/`
directories are ignored by Git and remain local unless deliberately archived elsewhere.

Per-token timings include Python API and timer overhead in both compared configurations.
Keep power settings stable, avoid heavy background work, retain slow samples, and report
the tokenizer, hardware, source commit, build flags, warmup count, timed repetitions, and
aggregation method with the results.

An unconstrained recursive-JSON workload previously exposed a cached-mask inconsistency:
one GPT-2 token was rejected by the generated mask although direct parser acceptance
allowed a valid continuation. `reproduce_adaptive_mask_mismatch.py` preserves the minimal
reproduction. Adaptive timing claims should be limited to workloads whose complete mask
traces agree between configurations.

References: [XGrammar 1 §4.3](https://arxiv.org/html/2411.15100v3#S4.SS3) and
[XGrammar 2 §4.1/§4.4](https://arxiv.org/html/2601.04426v4#S4).
