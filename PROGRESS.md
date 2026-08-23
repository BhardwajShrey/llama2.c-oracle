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

- **2026-08-23 — Phase 2 → 3 bridge: token embedding lookup.** Added
  `getTokenFloats`, which copies one token's row out of
  `w.tok_embeddings` (`dim` floats starting at `tokId * dim`) into a
  caller-provided `std::vector<float>`. This is the first read of the
  actual weight data by index rather than just walking pointers past it —
  the next real step of the forward pass (`x = tok_embeddings[token_id]`)
  now has a concrete function to build on. Also stubbed `dumpFloats` (not
  yet implemented — for writing intermediate activations to `dumps/` to
  diff against `oracle.py`'s output later) and commented out the earlier
  `advanced`/`printFirstN` debug prints now that the pointer math is
  trusted.

- **2026-08-19 — Phase 2 follow-up: wired up `Weights` pointers.** Walked
  a `float*` through the mapped region (right after the header) and
  assigned each `Weights` field an offset, in file order:
  `tok_embeddings → att_norm → wq → wk → wv → wo → ffn_norm → w1 → w2 → w3
  → final_norm`, then `output` (== `tok_embeddings` if `sharedWeights`,
  otherwise offset past two RoPE frequency tables that this loader doesn't
  keep pointers to). This closes out the byte-gap open question below.
  Added a `printFirstN` helper and spot-checked the first 5 floats of
  `tok_embeddings` and `wq`.

- **2026-08-16 — Phase 2 follow-up: fd/mmap cleanup + `Weights` struct
  scaffold.** Fixed the `fd` leak on a failed `fstat` (now `close(fd)`
  before returning) and added the missing `munmap(data, st.st_size)` on
  the successful exit path. Also declared a `Weights` struct (pointers for
  `tok_embeddings`, `att_norm`, `wq`/`wk`/`wv`/`wo`, `ffn_norm`,
  `w1`/`w2`/`w3`, `final_norm`, `output`) — not wired up to the mapped
  data yet, just the shape for the next step.

- **2026-08-16 — Phase 2 follow-up: switched header read from `fread` to
  `mmap`.** `main.cpp` now `open()`s the checkpoint, `fstat()`s it for
  size, and `mmap()`s the whole file read-only instead of using
  `FILE*`/`fread`, matching `run.c`'s approach (which mmaps everything
  past the header). `readConfig` now takes the mapped pointer and copies
  the header out via a cast instead of a stream read. Output on
  `stories15M.bin` unchanged from the `fread` version.

- **2026-08-15 — Phase 2 follow-up: byte-accounting sanity check.** Added
  a check in `main.cpp` that computes the expected weight-float count from
  the parsed `Config` (embedding table + all per-layer matrices/norms +
  final norm) and compares `expected bytes + header size` against the
  file's actual size (via `fseek`/`ftell`). On `stories15M.bin`: expected
  60,766,876 bytes, actual 60,816,028 — a 49,152-byte (12,288-float) gap.
  Not yet explained — see Open questions.

- **2026-08-13 — Phase 2 follow-up: hardened header parsing.** Switched the
  `fread` to read `sizeof(Config)` in one call instead of a hardcoded field
  count of `7`, so the read size stays tied to the struct instead of a
  magic number that can silently desync from it. Captured `shared_weights`
  from the sign of `vocab_size` before taking its absolute value, instead
  of discarding it. Struct blit works because `Config` is 7 `int32`s with
  no padding; a field-by-field reader is the portable version if the
  format ever crosses a machine boundary (different `int` size, different
  endianness, or padding introduced by a future struct change).

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
- ~~`stories15M.bin` is 12,288 floats (49,152 bytes) larger than
  `header + all layer weights + final norm` accounts for.~~ **Resolved
  2026-08-19:** it's two RoPE frequency tables (`seq_len × head_size/2`
  each, for the real/imaginary rotation components) that `run.c` maps but
  then never uses past computing an offset — this engine will compute RoPE
  angles directly instead of reading a cached table, so the classifier
  offset just skips past both (`seq_len * head_size` combined) without
  keeping pointers to them.

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
