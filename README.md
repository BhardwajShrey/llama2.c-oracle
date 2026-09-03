# llama2.c-oracle

This is my personal fork of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c), repurposed as a learning project. It is not trying to be a better or more complete version of the original — go there for the real thing. Everything below explains what I'm actually doing here.

## What I'm doing

I'm building a Llama 2 transformer **inference engine from scratch in C++**, then optimizing it, to learn low-level performance engineering — cache behavior, SIMD, quantization, memory bandwidth limits, that kind of thing. The optimization work is the actual point of the project. Correctness is table stakes, not the deliverable.

I'm not starting from a blank page. `run.c`/`runq.c` — Karpathy's original, dependency-free reference implementation — are being read, instrumented, and studied first, so I understand exactly what a correct, working transformer forward pass looks like in C before I write my own from scratch.

## How I'm going about it

Roughly, in phases (see [`PROGRESS.md`](PROGRESS.md) for the full checklist and a dated journal of what I did, learned, and broke at each step):

- **Phase 0 — baseline.** Ran `run.c` as-is, measured its throughput, instrumented `forward()`/`matmul()` to confirm where the time actually goes before optimizing anything.
- **Phase 1 — oracle & comparison tooling.** Built [`oracle.py`](oracle.py) (dumps ground-truth per-layer activations from the PyTorch reference model) and [`compare.py`](compare.py) (diffs my engine's output against those dumps within a numeric tolerance). Every correctness claim about the from-scratch engine has to survive this comparison.
- **Phase 2 — C++ skeleton.** [`main.cpp`](main.cpp): parse the `.bin` checkpoint header, memory-map the weights.
- **Phase 3 — forward pass** *(in progress)*. Implement RMSNorm, attention, RoPE, SwiGLU — one piece at a time, checked against the oracle after each one.
- **Phase 4 — optimization.** SIMD, cache-aware matmul, quantization, threading. Every performance-affecting change gets a logged before/after measurement — see [`BENCHMARKS.md`](BENCHMARKS.md) for the protocol and the log itself.
- **Phase 5 — real model.** Run something bigger than the tiny TinyStories checkpoints through the finished engine.

[`GLOSSARY.md`](GLOSSARY.md) is a plain-language reference I keep open while working — what each weight matrix (`wq`, `wk`, `w1`, `w2`, `w3`, ...) actually is, what terms like RoPE/KV-cache/quantization mean, and the exact config for the `stories15M` model I'm using to develop against.

## How I'm using Claude

I use Claude Code as a guide and reviewer while doing this, not as the person writing the engine. The ground rule, enforced via [`CLAUDE.md`](CLAUDE.md): **Claude does not write inference engine code** — no matmul, no SIMD intrinsics, no quantization, no attention logic. That's the entire point of the project, and having Claude write it would defeat it.

What Claude *does* do:
- writes the tooling around the engine — `oracle.py`, `compare.py`, build/test scripts
- writes and maintains this project's documentation (this file, `GLOSSARY.md`, `PROGRESS.md`, `BENCHMARKS.md`)
- explains concepts on request (cache behavior, SIMD, quantization theory, transformer internals)
- reviews every commit that touches `main.cpp` for bugs, undefined behavior, missing error handling, and non-idiomatic C++ — and explains *why* each one matters — without rewriting the code or handing back a fix. I fix what it flags myself.

## Files I've added on top of the original repo

Everything else in this repo (`run.c`, `runq.c`, `model.py`, `train.py`, `export.py`, etc.) is Karpathy's original reference implementation — largely untouched, aside from some timing instrumentation added inside `run.c`'s `forward()`/`matmul()` for Phase 0. These are what I added:

| File | Purpose |
| --- | --- |
| [`main.cpp`](main.cpp) | The from-scratch C++ inference engine — the actual deliverable of this project, built up incrementally |
| [`CLAUDE.md`](CLAUDE.md) | Rules for how Claude Code should (and shouldn't) work in this repo |
| [`GLOSSARY.md`](GLOSSARY.md) | Plain-language reference for transformer terminology and weight-matrix naming |
| [`PROGRESS.md`](PROGRESS.md) | Phase checklist plus a dated running journal of the work |
| [`BENCHMARKS.md`](BENCHMARKS.md) | Performance measurement protocol and the log of every benchmark taken |
| [`oracle.py`](oracle.py) | Dumps per-layer PyTorch activations to `dumps/*.npy`, via forward hooks — the correctness ground truth |
| [`compare.py`](compare.py) | Diffs an oracle `.npy` dump against a raw float32 dump from the C++ engine |
| `.gitignore` | Ignores the venv, build artifacts, checkpoints, and dump output this workflow generates |

## Building and running the original reference implementation

The upstream `run.c`/`runq.c`/training pipeline still works exactly as documented in the original project — see [`CLAUDE.md`](CLAUDE.md) for the exact commands (build targets, running a checkpoint, training, exporting, testing, the Python scripts). For the full original writeup — the "feel the magic" quick start, Meta Llama 2 weight conversion, int8 quantization background, training guide, custom tokenizers, and the huge list of ports to other languages — see the upstream repo: **https://github.com/karpathy/llama2.c**.

## License

MIT (unchanged from upstream).
