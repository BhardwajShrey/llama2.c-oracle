# Progress

Running journal for the llama2.c-from-scratch learning project. Also a record of how Shrey repeatedly makes a fool out of himself while doing this.

## Status

Phase 2 (C++ skeleton: parses the `.bin` header, prints config) is done;
Phase 3 (forward pass) is essentially complete for a single token/position
— embedding → 6 layers (attention + SwiGLU FFN) → final RMSNorm →
classifier matmul now runs end to end, and its argmax/top logit
(`9038`, `12.29`) match `oracle.py`'s PyTorch output exactly for token
1 / pos 0. GQA assumptions (`kv_dim` vs `dim`) are still open per the
TODOs in `main.cpp`; multi-position generation (KV cache reuse across a
real sequence) hasn't been exercised yet.

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
- [ ] **Phase 3 (in progress) — forward pass.** Implement the actual
      transformer forward pass in the from-scratch C++ engine, validate
      against the oracle. Full single-token forward pass (embedding
      through classifier) validated against `oracle.py` — argmax and top
      logit match exactly. GQA assumptions and multi-position generation
      remain before this phase is fully done.
- [ ] **Phase 4 — optimization.** SIMD, cache-aware matmul, quantization,
      threading — the actual point of the project. Every change measured
      before/after per `BENCHMARKS.md`.
- [ ] **Phase 5 — real model.** Run something bigger than stories15M/42M
      through the finished engine.

## Log

*Reverse-chronological — newest entry first.*

- **2026-09-23 — Phase 3: final RMSNorm + classifier head — full forward
  pass validated end to end.** After the layer loop, added
  `rmsNorm(s.x.data(), w.final_norm, config.dim, s.x.data())` (aliasing
  `x`/`out` deliberately — safe here since `rmsNorm`'s first pass reads
  all of `x` before the second pass writes it, unlike `matmul`, which the
  code's own disclaimer warns against aliasing) followed by
  `matmul(s.logits.data(), s.x.data(), w.output, config.dim,
  config.vocab_size)` — matches `run.c`'s `rmsnorm(x, x,
  w->rms_final_weight, dim)` + `matmul(s->logits, x, w->wcls, dim,
  vocab_size)` exactly. `oracle.py` got a matching two-line addition
  printing `np.argmax`/`.max()` of `dumps/logits.npy`. **Verified: the
  C++ engine's argmax and top logit value (`9038`, `12.29`) match the
  PyTorch oracle's (`9038`, `12.2900305`) exactly** for token 1 / pos 0
  — the entire forward pass (embedding → 6 layers → final norm →
  classifier) is now confirmed numerically correct end to end, checked
  via a temporary print added to a scratch build (not committed —
  `main.cpp` computes `maxIndex` but doesn't print it yet, so this
  result isn't visible just from running the checked-in binary).

- **2026-09-21 — Phase 3 follow-up: layer-indexed dump filenames.**
  Every `dumpFloats` call inside the layer loop now builds its path with
  a `layerSuffix` (`"_layer_" + std::to_string(layer) + ".bin"`) instead
  of a fixed name — `mine/att_norm_layer_0.bin` … `_layer_5.bin`, and
  likewise for `matmul_wq/wk/wv`, `att_xb`/`att_xb2`, and
  `ffn_norm`/`ffn_w1`/`ffn_w2`/`ffn_w3`. Fixes the dump-clobbering issue
  found earlier today: verified the new `_layer_5` file for each dump
  point is byte-identical to what the old fixed-filename version left
  behind (since layer 5 ran last and always won the overwrite),
  confirming this change only affects where output lands, not what's
  computed. `<string>` added to includes rather than relying on
  transitive inclusion.

- **2026-09-21 — Phase 3: bumped the layer loop to `config.n_layers`.**
  Second half of the two-step plan: `int layer {0};` became
  `for (int layer = 0; layer < config.n_layers; layer++)`, wrapping the
  same body with no other changes — confirmed via a whitespace-ignoring
  diff against the previous commit that nothing inside the loop was
  altered, only the declaration/wrapper. Builds clean, runs to
  completion with no NaNs (spot-checked `att_norm.bin`). **Found while
  reviewing:** every `dumpFloats` call inside the loop targets a fixed
  filename (`mine/att_norm.bin`, etc.), so across 6 iterations each file
  gets overwritten 6 times — the file left on disk after the run holds
  layer 5's values, not layer 0's, silently breaking the correspondence
  with `oracle.py`'s per-layer dumps (`dumps/layers.0.*.npy`,
  `dumps/layers.1.*.npy`, ...). `s.x` itself is unaffected — it correctly
  carries state across all 6 layers — this only breaks the *intermediate*
  dumps used for oracle comparison. Not yet fixed.

- **2026-09-21 — Phase 3: extracted `forward()`.** Pulled the entire
  embedding-lookup + single-layer body out of `main()` into
  `forward(RunState& s, const Config& config, const Weights& w, int
  token_id, int pos)`; `main()` now shrinks to setup, one `forward()`
  call, and cleanup. `layer` is still a fixed local (`int layer {0};`),
  not a loop — this commit is the extraction only, kept deliberately
  separate from bumping the loop bound to `config.n_layers` so each step
  can be verified independently. Verified all 11 `mine/*.bin` dumps
  byte-identical against the previous commit's output before committing.
  (A previous attempt combined the extraction and the loop-bound bump in
  one uncommitted change and silently dropped the final-norm/classifier
  step the plan called for; that attempt was discarded via local
  `git reset --hard` before it was ever pushed.)

- **2026-09-20 — Phase 3 follow-up: fixed `wo`'s missing layer offset,
  comment/param fixes.** `w.wo`'s matmul now uses `+ kqvOffset`, same
  stride as `wq`/`wk`/`wv` (matches `run.c`'s `w->wo + l*dim*dim`) —
  closes the gap flagged 2026-09-19. Also fixed `matmul`'s comment (was
  backwards: it had `d`/`n` swapped relative to what the code actually
  does) and `swiGLU` now writes through its `out` parameter instead of
  silently mutating `hb` regardless of what's passed as `out` — safe
  since the call site aliases `out`/`hb` to the same buffer, but the
  function itself is honest about it now. All dumps still byte-identical
  to before, as expected at `layer=0` (`kqvOffset` is `0`).

- **2026-09-19 — Phase 3: `wo` residual fix, SwiGLU FFN, per-layer
  weight offsets, eps moved inside `rmsNorm`.** Fixed a real bug: the
  post-attention residual add was `s.x[i] += s.q[i]` (adding the raw
  query vector back into the residual stream instead of the actual
  `wo`-projected attention output, `s.xb2`) — now correctly
  `s.x[i] += s.xb2[i]`. Added the SwiGLU feed-forward block (`w1`/`w3`
  matmuls, `swiGLU()` combining them via `sigmoid`, `w2` matmul back
  down, residual add), matching `run.c`'s FFN section exactly. Added
  explicit per-layer offsets (`kqvOffset`, `w1w3offset`, `w2offset`) to
  every weight lookup that needed one (`att_norm`, `wq`/`wk`/`wv`,
  `ffn_norm`, `w1`/`w2`/`w3`) — all checked against `run.c`'s equivalent
  offset expressions and correct. **Found while checking:** `w.wo`'s
  matmul call has no layer offset at all, unlike every other per-layer
  weight — invisible today since `layer` is hardcoded to `0`, but it'll
  break the moment the layer loop exists. At my suggestion (and with the
  user's explicit go-ahead to override the "no engine code" rule for
  this one case), also pulled `eps` (`1e-5`) inside `rmsNorm` itself
  instead of passing it from every call site, and updated the comment
  above the function to match — verified byte-identical dumps before and
  after.

- **2026-09-18 — Phase 3 follow-up: extracted `writeToCache`.** Pulled
  the two `std::copy` calls writing `k`/`v` into their caches into a
  single `writeToCache(in, cache, layer, pos, config)` helper. No
  behavior change — still goes through `cacheOffset`, so the existing
  `kv_dim`-vs-`dim` TODO still applies here too.

- **2026-09-17 — Phase 3 follow-up: fixed the softmax off-by-one.**
  Normalization loop now runs `i <= pos` (was `i < pos`), matching
  `run.c`'s `softmax(att, pos+1)`. Also made `dot()` take `const float*`
  for both inputs, matching `matmul`'s style. Since `pos=0` is still the
  only tested case, this fix is invisible in today's output — the bug it
  fixes only manifests once there's more than one cached position — but
  the code is now correct ahead of that, rather than accidentally correct
  because of the test's current shape.

- **2026-09-17 — Phase 3: attention score, softmax, weighted sum.**
  Implemented the rest of attention per head: dot-product score against
  every cached position `0..pos` (scaled by `1/sqrt(head_dim)`), softmax
  over those scores, then a weighted sum of the cached values into
  `s.xb`, dumped to `mine/att_xb.bin`. Matches `run.c`'s per-head loop
  structure and its `dot`/scale/softmax/weighted-sum steps. **Found a
  bug while reviewing:** the softmax normalization pass looped `i < pos`
  instead of `i <= pos` — fixed same day, see follow-up entry above.

- **2026-09-15 — Phase 3: KV cache, first entries written.** Added
  `key_cache`/`value_cache` (`n_layers * seq_len * dim` each) to
  `RunState`, and `cacheOffset(layer, pos, config)` to compute where a
  given layer/position's slice starts. After RoPE, copies the current
  (post-rotation) `k` and (raw) `v` into the cache at layer 0, position 0
  — mirrors `run.c`'s pattern of storing rotated keys and unrotated
  values. Attention's actual score/softmax/weighted-sum step is next; the
  cache exists now so that step has somewhere to read previous positions'
  K/V from once there's more than one position.

- **2026-09-06 — Phase 3: RoPE, via the checkpoint's precomputed tables
  (correction to the 2026-08-19 plan).** Added `w.cos_table`/`w.sin_table`
  to `Weights`, pointing at the two `seq_len × (head_dim/2)` blocks
  `initWeights` was previously just skipping past, and a `rope()` that
  rotates `q`/`k` (never `v`) pair-by-pair per head using those tables,
  matching `run.c`'s rotation formula (`v0*cos - v1*sin`,
  `v0*sin + v1*cos`) and `model.py`'s adjacent-pair convention. **This
  reverses what the 2026-08-19 entry said** ("this engine will compute
  RoPE angles directly instead of reading a cached table") — turns out
  the checkpoint's legacy export format (`export.py`'s `legacy_export`)
  really does write real, usable `freqs_cos`/`freqs_sin` tables (`run.c`
  just chooses to recompute them at runtime via `cosf`/`sinf` instead of
  reading them back); reading them directly is simpler here and the two
  approaches are mathematically identical.

- **2026-09-05 — Phase 3: introduced `RunState`.** Replaced the growing
  pile of one-off `std::vector<float>` locals in `main` (`x`, `rmsOut`,
  `matmulWq`, `matmulWk`, `matmulWv`) with a `RunState` struct
  (`x`, `xb`, `q`, `k`, `v` so far, with `hb`/`hb2`/`att`/`logits`/KV-cache
  fields commented in as placeholders for what's coming), built once via
  `createRunState(config)`. Mirrors `run.c`'s own `RunState`/
  `malloc_run_state` split between config-derived buffers and the
  weights. Also tightened `rmsNorm` to take `float* out` (matching `x`,
  which switched last commit — closes the inconsistency flagged then) and
  cast `n` to `size_t` before the `i * n` multiply in `matmul`, ahead of
  it mattering on a bigger model.

- **2026-09-05 — Phase 3: Q/K/V projections.** Extended the single
  `matmul` call against `w.wq` to also run against `w.wk` and `w.wv`, all
  three against the same RMSNorm output — `matmulWq`/`matmulWk`/`matmulWv`,
  dumped to `mine/matmul_w{q,k,v}.bin`. This is the query/key/value step
  of attention (still layer 0, still token 1, still no RoPE or the actual
  attention score/softmax/weighted-sum yet — just the three linear
  projections that feed into it).

- **2026-09-05 — Phase 3 follow-up: `getTokEmbedding`/`rmsNorm` take raw
  pointers instead of `std::vector&`.** Both functions now take `float* x`
  plus an explicit `dim` parameter instead of a `std::vector<float>&`,
  with call sites passing `x.data()`. Removes the implicit assumption that
  the caller's vector happens to be sized `dim` — size is now a parameter,
  not inferred from `x.size()`.

- **2026-09-05 — Phase 3: first `matmul`.** Implemented a plain scalar
  `matmul(out, x, w, n, d)`: a `d × n` weight matrix times an `n × 1`
  vector, giving a `d × 1` output — the "matrix times *vector*, one token
  at a time" shape noted in `GLOSSARY.md`, not a general matrix-matrix
  multiply. Ran it on the RMSNorm output against `w.wq` (still layer 0
  only) and dumped the result to `mine/matmul_wq.bin`. This is the
  reference/correctness version — Phase 4 is where this function gets
  rewritten for speed (SIMD, cache blocking, etc.), so its current form is
  deliberately the simplest thing that's obviously correct.

- **2026-09-04 — Phase 3: RMSNorm.** Implemented `rmsNorm` (sum of
  squares over `dim` → divide by `dim` → add `eps` → reciprocal square
  root → scale each element by that and by the corresponding norm
  weight), matching `run.c`'s `rmsnorm()` formula and its hardcoded
  `1e-5f` epsilon (not stored in `Config` — same in both the reference
  C and `model.py`). Applied it to the token-1 embedding using
  `w.att_norm` (layer 0's slice) and dumped the result to
  `mine/att_norm.bin` for comparison against `oracle.py`'s
  `layers.0.attention_norm.npy`. This is the first actual forward-pass
  computation in the from-scratch engine — everything before this was
  header parsing and pointer setup.

- **2026-08-24 — Phase 2 → 3 bridge: implemented `dumpFloats`, first
  real oracle comparison.** `dumpFloats` now writes a raw float32 array to
  disk (`fopen`/`fwrite`/`fclose`, checked for a short write), replacing
  the earlier stub. Wired it into `main`: looks up token id `1`'s
  embedding row and dumps it to `mine/embeddings.bin`. This is the first
  C++-side output that's actually in the format `compare.py` expects —
  next step is running `compare.py dumps/tok_embeddings.npy
  mine/embeddings.bin` (once `oracle.py` dumps a per-token embedding, not
  just full-sequence tensors) to confirm the lookup matches PyTorch.
  `mine/` added to `.gitignore` as the output directory for these dumps.

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
  2026-08-19, corrected 2026-09-06:** it's two RoPE frequency tables
  (`seq_len × head_size/2` each). The 2026-08-19 note said this engine
  would skip them and recompute RoPE angles on the fly like `run.c`
  does — instead, as of 2026-09-06, `initWeights` keeps pointers to both
  (`w.cos_table`/`w.sin_table`) and `rope()` reads them directly.
- `expected_file_size()` still doesn't count `cos_table`/`sin_table` in
  its byte total, even though `initWeights` now walks past and uses both
  — so the "gap" the program prints at startup still reports 12,288
  floats unaccounted for, when actually every float in the file is now
  spoken for. The diagnostic just hasn't been updated to match; the two
  functions computing checkpoint layout size have desynced the same way
  the per-field size formulas did before (same root cause, same fix:
  don't keep the same math in two places).

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
