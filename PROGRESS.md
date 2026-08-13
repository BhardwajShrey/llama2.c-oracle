# Progress

Running journal for the llama2.c-from-scratch learning project. Also a record of how Shrey repeatedly makes a fool out of himself while doing this.

## Status

Phase 2 (C++ skeleton: parses the `.bin` header, prints config) is done;
Phase 3 (forward pass) is next.

## Phase checklist

- [x] **Phase 0 — baseline.** Ran `llama2.c` as-is, measured throughput:
      110 tok/s (`-O3`) vs. 670 tok/s (`-Ofast -march=native`). Instrumented
      `forward()` and `matmul()` in `run.c` and confirmed `forward()`
      dominates runtime.
- [x] **Phase 1 — oracle & comparison tooling.** Built `oracle.py`
      (PyTorch reference model, forward hooks, dumps 64 tensors to
      `dumps/`). Built `compare.py` (1e-4 relative tolerance), verified it
      actually catches a deliberate corruption.
- [x] **Phase 2 — C++ skeleton.** Parse the `.bin` checkpoint header, print
      the config.
- [ ] **Phase 3 (next) — forward pass.** Implement the actual transformer
      forward pass in the from-scratch C++ engine, validate against the
      oracle.
- [ ] **Phase 4 — optimization.** SIMD, cache-aware matmul, quantization,
      threading — the actual point of the project. Every change measured
      before/after per `BENCHMARKS.md`.
- [ ] **Phase 5 — real model.** Run something bigger than stories15M/42M
      through the finished engine.

## Log

*Reverse-chronological — newest entry first.*

- **2026-08-13 — Phase 2: `main.cpp` skeleton.** Wrote the `Config` struct
  matching the `.bin` header layout (7 `int32`s: `dim`, `hidden_dim`,
  `n_layers`, `n_heads`, `n_kv_heads`, `vocab_size`, `seq_len`), opened
  `stories15M.bin` and `fread`'d the header straight into it, printed every
  field to confirm it matches the known stories15M config. `vocab_size`
  prints as negative in the raw header (sign is a flag elsewhere in the
  original format) — printed with `abs()` for a sane-looking number.

- **2026-08-08 — Phase 1: `compare.py`.** Wrote the comparison script:
  loads a `.npy` (oracle) and a raw float32 dump (C++ side), diffs them,
  reports max absolute/relative error and the worst-offending index.
  Deliberately corrupted a dumped value to confirm the 1e-4 relative
  tolerance actually catches it before trusting it for real comparisons.

- **2026-08-08 — Phase 1: `oracle.py`.** Loaded the `stories15M`
  checkpoint into the PyTorch reference model, registered a forward hook on
  every named submodule, ran one forward pass, dumped 64 tensors to
  `dumps/*.npy` (per-layer attention/feed-forward/norm outputs, plus
  embeddings, final norm, and logits). This is now the ground truth the
  C++ engine will be checked against.

- **2026-08-07 — Phase 0: writeup.** Logged the `-O3` vs. `-Ofast
  -march=native` throughput numbers in `BENCHMARKS.md` following the
  before/after protocol. Confirmed via instrumentation that `forward()` —
  not tokenization, not sampling — is where essentially all runtime goes.

- **2026-08-05 — Phase 0: instrumentation.** Added timing instrumentation
  around `forward()` and `matmul()` in `run.c`. Built and ran `stories15M`
  under both `-O3` and `-Ofast -march=native` to get a baseline before
  touching anything.

## Open questions

- Is the `-Ofast -march=native` build already memory-bandwidth-bound?
  670 tok/s × 60.8 MB (checkpoint size) ≈ 40 GB/s of weight traffic — needs
  to be measured against this machine's actual peak memory bandwidth before
  assuming there's compute headroom left to optimize.

## Gotchas hit

- `load_state_dict(strict=False)` will **silently load nothing** if the
  `_orig_mod.` prefix (left over from `torch.compile`) isn't stripped from
  the checkpoint's keys first — no error, just a model full of its random
  init weights. Always load with `strict=True` so a mismatch fails loudly
  instead of failing silently.
- `-Ofast` implies `-ffast-math`, which reorders floating-point additions
  and changes results slightly (not a bug, just non-bit-exact). Use `-O2`
  when checking correctness against the oracle, and only switch to
  `-Ofast` for performance runs.
- `requirements.txt` pins `torch==2.0.1`, from 2023 — do not actually
  install that. Install packages reactively, only when a
  `ModuleNotFoundError` demands it.
- `sample.py` uses a fixed seed (1337). Increasing `max_new_tokens`
  continues the *same* story rather than generating a new one — don't
  mistake that for the model repeating itself.
