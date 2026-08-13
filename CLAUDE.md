# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is a personal fork of karpathy/llama2.c, repurposed as a learning project: **building a transformer inference engine from scratch in C++, then optimizing it, to learn low-level performance engineering.** The upstream `run.c`/`runq.c` reference implementation is being instrumented and studied first, before the from-scratch engine is written. The optimization work *is* the point of this project — correctness is table stakes, not the deliverable.

Three project docs track this work and are kept up to date as it progresses: `GLOSSARY.md` (term/weight-matrix reference), `PROGRESS.md` (phase checklist + dated log), and `BENCHMARKS.md` (perf measurement log, see "Benchmark discipline" below).

## Hard rule: you do not write engine code here

Do **not** write inference engine code, matmul implementations, SIMD intrinsics, quantization code, or attention logic. The user writes all of that themselves — that's the entire point of the project. You may:

- write build scripts, test harnesses, and comparison/profiling tooling
- write documentation (like this file and `BENCHMARKS.md`)
- explain concepts (cache behavior, SIMD, quantization theory, etc.)
- review code the user has written

If asked to implement core engine logic (matmul, SIMD, quantization, attention, etc.), **remind the user of this rule and ask whether they want to override it** rather than just writing it.

## Code review on every commit

Whenever the user commits engine code they wrote themselves (`main.cpp` and any future from-scratch engine files), review `git show HEAD` (or the relevant diff) for bugs, undefined behavior, missing/weak error handling, and non-idiomatic C++ — and explain *why* each one matters, tying it back to this project's own context (the file format, `run.c`'s reference behavior, GLOSSARY.md concepts, etc.) where relevant. **Do not rewrite the code or hand back a corrected version** — point out the issue and let the user fix it themselves, same as the hard rule above. Do this proactively after any commit that touches engine code, not just when asked.

## Benchmark discipline

Any performance-affecting change must be measured before and after using the protocol documented at the top of `BENCHMARKS.md` (deterministic sampling, fixed token count, exact command + build flags recorded), and the result logged there. A perf change without a logged before/after measurement isn't done.

## Determinism

All correctness comparisons between implementations (reference vs. from-scratch, `-O3` vs `-Ofast`, etc.) use `-t 0` (deterministic/greedy sampling). Never validate correctness against stochastic (temperature > 0) output — you can't tell a real bug from sampling noise that way.

## Build and run (C inference engine)

```bash
make run              # builds ./run and ./runq with -O3
make runfast           # -Ofast, faster but breaks strict IEEE/C compliance
make runomp            # -Ofast -fopenmp -march=native, multithreaded (needs clang+libomp on Mac: `make runomp CC=/opt/homebrew/opt/llvm/bin/clang`)
make rungnu             # -std=gnu11, for CentOS7/Amazon Linux
make runompgnu          # OpenMP + gnu11
make win64               # cross-compile Windows .exe via mingw
make rundebug             # -g build for valgrind/debugging
make clean
```

Run a model checkpoint:

```bash
./run stories15M.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth"   # -t temp, -p top-p, -n steps, -i prompt, -m chat|generate, -z custom tokenizer.bin
./runq model_q80.bin       # int8-quantized checkpoint (must be exported with --version 2)
OMP_NUM_THREADS=4 ./run out/model.bin   # when built with OpenMP
```

## Tests

```bash
pytest                  # runs test_all.py: forwards a model through both run.c and PyTorch for 200 steps, diffs against a known-good output
make testc               # same, but only the run.c-based test (faster if only C changed)
make testcc VERBOSITY=1   # builds and runs test.c (C-level tokenizer unit tests), VERBOSITY=1 for verbose output
```

`test_all.py` downloads the tiny `stories260K` checkpoint + `tok512` tokenizer into `test/` on first run (~2MB). There is no single-test filter beyond `pytest -k runc` / `pytest -k python` (the two test functions in `test_all.py` are `test_runc` and `test_python`).

## Python environment

A venv already exists at `.venv/` (Python 3.9). Activate it before running any Python script:

```bash
source .venv/bin/activate
```

`pytest` is listed in `requirements.txt` but is **not currently installed** in `.venv` — run `pip install pytest` (or `pip install -r requirements.txt`) inside the activated venv before `pytest`/`make testc` will work.

```bash
pip install -r requirements.txt   # only needed once, or after requirements.txt changes
```

Key deps: `torch==2.0.1` (installed as 2.8.0 in `.venv`), `sentencepiece` (custom tokenizer training), `wandb` (optional logging), `tqdm`, `requests`.

### Running the Python scripts

Most scripts here use the "poor man's configurator" pattern (`exec(open('configurator.py').read())`): plain module-level variables at the top of the file act as defaults, and you override them from the CLI with `--key=value`. A few (`export.py`, `tokenizer.py`, `tinystories.py`) use standard `argparse` instead. Always run these with the venv activated.

```bash
# sample.py — pure-PyTorch inference, cross-check against run.c/runq.c output
python sample.py --checkpoint=out/stories15M.pt --max_new_tokens=100 --seed=1337
# temperature=0 sampling for deterministic comparisons:
python sample.py --checkpoint=out/stories15M.pt --temperature=0.0

# oracle.py — dumps per-layer activations from the PyTorch reference model to dumps/*.npy
# (edit checkpoint_path at the top of the file — it's not configurator-driven)
python oracle.py

# compare.py — diffs a PyTorch .npy dump against a raw float32 dump from the C++ engine
python compare.py dumps/logits.npy out/my_cpp_dump.bin

# train.py — trains model.py's Transformer (configurator-driven, can also take a config file)
python train.py config/tinystories15M.py --batch_size=32

# export.py — converts a .pt checkpoint to the .bin format run.c/runq.c read (argparse-driven)
python export.py out/stories15M.bin --checkpoint=out/stories15M.pt --version=1
python export.py out/stories15M_q80.bin --checkpoint=out/stories15M.pt --version=2   # int8, for runq.c

# tinystories.py — dataset download/pretokenization/vocab training (argparse-driven, positional stage)
python tinystories.py download
python tinystories.py pretokenize
python tinystories.py train_vocab --vocab_size=512

# tokenizer.py — export a trained sentencepiece .model into the tokenizer.bin format run.c reads
python tokenizer.py -t data/tok512.model

# test_all.py — see Tests section above; needs pytest installed in .venv first
pytest
```

## Architecture

### Two independent implementations of the same model, kept numerically identical

- **`model.py`** — the reference PyTorch `nn.Module` (`Transformer`, `ModelArgs`) used for training.
- **`run.c`** — a from-scratch, dependency-free re-implementation of forward-pass inference (RoPE, RMSNorm, multi-head/grouped-query attention, SwiGLU MLP) reading raw weights via `mmap`. There is no shared code between them — architecture changes must be ported to *both* by hand, and `pytest` (`test_all.py`) is what proves they still agree bit-for-bit at temperature 0.
- **`runq.c`** — same as `run.c` but for int8 (Q8_0) quantized checkpoints exported with `--version 2`; weights are quantized, activations are dynamically quantized/dequantized at runtime for matmuls.

### Pipeline: train → export → infer

1. **`tinystories.py`** — downloads/pretokenizes the TinyStories dataset into `data/` (or trains + uses a custom `sentencepiece` vocab via `train_vocab`/`--vocab_size`). Exposes `Task.iter_batches` used by `train.py`.
2. **`train.py`** — trains `model.py`'s `Transformer` (supports single-GPU and DDP via `torchrun`). Hyperparameters are plain module-level globals overridden through **`configurator.py`**'s "poor man's configurator" (`python train.py config/foo.py --batch_size=32` — exec's a config file, then applies `--key=value` overrides via `literal_eval` against `globals()`). Checkpoints go to `out_dir`; `train.py` calls `export.py`'s `model_export` to also write the `.bin` inference format at each checkpoint.
3. **`export.py`** — converts PyTorch checkpoints (this repo's own, Meta's official Llama 2 weights via `--meta-llama`, or HF weights via `--hf`) into the `.bin` format `run.c`/`runq.c` read. Supports multiple output versions: `v0` (legacy, no header), `v1+` (proper header, cache-aligned), `v2` (int8 Q8_0 quantized, for `runq.c`). `model_export(model, filepath, version, dtype)` is the dispatch point.
4. **`sample.py`** — pure-PyTorch inference (no C), used as a cross-check against `run.c`/`runq.c` output and in `test_all.py`.
5. **`tokenizer.py`** — wraps `sentencepiece`; `Tokenizer` class for Python-side encode/decode, plus a CLI to export a trained `.model` into the `.bin` format `run.c`'s `build_tokenizer` reads (`tokenizer.bin` is the default 32K Llama 2 vocab; custom vocabs live under `data/tok<N>.{model,bin}`).

### run.c internal structure (single file, ~900 lines)

Reading top to bottom: `Config`/`TransformerWeights`/`RunState`/`Transformer` structs → `build_transformer`/`memory_map_weights` (mmaps the `.bin` checkpoint directly into weight pointers, no deserialization) → `forward()` (the actual transformer forward pass) → `Tokenizer` struct + `build_tokenizer`/`encode` (BPE) → `Sampler` + `sample_*` functions (greedy/top-p/temperature) → `generate()`/`chat()` (the two run modes, selected via `-m`) → `main()` (CLI arg parsing, see `error_usage()` for all flags). `runq.c` mirrors this structure with `QuantizedTensor` and quantize/dequantize helpers added.

`test.c` builds by `#define TESTING` + `#include "run.c"`, giving it direct access to `run.c`'s static functions (currently just tokenizer encoding tests).

## Upstream reference material

The sections above (Build/run, Tests, Python environment, Architecture) describe the original karpathy/llama2.c project as it exists in this fork — still accurate and useful for understanding the reference implementation being studied. Upstream's own contributing philosophy (keep `run.c` minimal and mergeable) doesn't apply here since this fork isn't tracking upstream merges; the rules in "Hard rule," "Benchmark discipline," and "Determinism" above are what govern this project instead.
