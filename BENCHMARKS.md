# BENCHMARKS.md

Permanent running log of every performance measurement taken in this project. Append, don't overwrite — old entries stay even after they're superseded, so the log shows the trajectory.

## Measurement protocol

- All benchmarks use **deterministic sampling** (`-t 0`) and a **fixed token count** (`-n 256`), so runs are directly comparable to each other. Never log a tok/s number produced with temperature > 0 or a different `-n`.
- Every entry must record the **exact command** (including model path and flags) and the **exact build flags** used to compile the binary that ran it. "I rebuilt with -Ofast" is not enough detail — copy the literal `gcc`/`clang` invocation.
- Prefer running each measurement a couple of times and logging a representative number in Notes if there's significant run-to-run variance; note the variance if it's large.
- A benchmark entry is a measurement, not a vibe — if you didn't run the command yourself with this build, don't log it.

## Hardware

Fill these in once and update if you switch machines.

- **CPU model:** Apple M1 (MacBook Air, 2020)
- **Core count (physical / logical):** 8 / 8 (4 performance + 4 efficiency)
- **RAM:** 8 GB (unified memory, shared with GPU)
- **Memory bandwidth:** 68.25 GB/s (published Apple spec for M1's LPDDR4X-4266 over a 128-bit bus — not independently measured on this machine)
- **OS:** macOS 26.3.1 (build 25D771280a)

Note: the MacBook Air is fanless (passive cooling only), unlike the MacBook Pro's M1. Under sustained load — e.g. a long benchmark run, or several back-to-back runs — this machine can thermally throttle, which will silently lower tok/s and make runs look like a regression that isn't one. If a number looks off, let the machine cool and re-run before trusting it, and prefer noting ambient conditions in Notes for any run that felt hot.

## Results

| Date | Build flags | Command | Model | Tokens | tok/s | Delta vs previous | Notes |
|---|---|---|---|---|---|---|---|
| _needs re-measurement_ | `gcc -O3 -o run run.c -lm` | `./run stories15M.bin` | stories15M | ~256 (non-deterministic run) | ~110 | — | Original figure from README, not from this project's own measurements. Re-measure under `-t 0 -n 256` protocol before trusting for comparisons. |
| _needs re-measurement_ | `gcc -Ofast -march=native -o run run.c -lm` | `./run stories15M.bin` | stories15M | ~256 (non-deterministic run) | ~670 | ~6.1x vs -O3 row above (approximate, not apples-to-apples) | Same caveat: not measured under the deterministic protocol yet. `-march=native` means this number is tied to the CPU it was measured on — re-check the Hardware section matches before trusting it. |

## Findings

- `forward()` dominates total runtime — essentially all optimization effort belongs inside `matmul()`. Everything else in the forward pass (rmsnorm, softmax, RoPE, attention bookkeeping) is comparatively cheap.
- `-Ofast` implies `-ffast-math`, which permits floating-point reassociation (and other IEEE-754-breaking transforms). This means `-Ofast` builds can produce slightly different numerical output than `-O3` builds even at `-t 0` — don't treat a diff between an `-O3` run and an `-Ofast` run as a correctness bug without accounting for this first.
