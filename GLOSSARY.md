# Glossary

Reference doc for the llama2.c-from-scratch learning project. Written for a
C++/systems background with zero assumed ML background.

## Naming convention

Every per-layer weight matrix is named `w<something>` — `wq`, `wk`, `wv`,
`wo`, `w1`, `w2`, `w3`. The `w` just means "weight matrix" (as opposed to a
bias, a norm parameter, etc). The rest of the name (`q`, `k`, `v`, `o`, or a
bare number) comes straight from the notation used in the original
transformer paper ("Attention Is All You Need") and the papers that followed
it — it is **not** descriptive of what the matrix does. You have to already
know what `q`/`k`/`v` mean, or look it up, the name alone won't tell you.

## Per-layer weights (7 matrices)

Every layer has the same 7 weight matrices: 4 for attention, 3 for the
feed-forward block.

| Name | Full name | Shape (stories15M) | Plain-English purpose | Execution order |
|---|---|---|---|---|
| `wq` | query weight | 288×288 | Projects the layer's input into a "query" vector — what this token is looking for | 1st (together with `wk`, `wv`) |
| `wk` | key weight | 288×288 | Projects the input into a "key" vector — what this token offers, to be matched against other tokens' queries | 1st (together with `wq`, `wv`) |
| `wv` | value weight | 288×288 | Projects the input into a "value" vector — the actual content that gets pulled forward when attention matches | 1st (together with `wq`, `wk`) |
| `wo` | output weight | 288×288 | Mixes the attention heads' outputs back into one vector, which gets added into the residual stream | 2nd |
| `w1` | gate weight | 288 → 768 | Expands the vector to `hidden_dim` and produces the "gate" signal (passed through SiLU) | 3rd (together with `w3`) |
| `w2` | down weight | 768 → 288 | Compresses the gated result back down to `dim` — this **is** the feed-forward block's output | 4th, **LAST** |
| `w3` | up weight | 288 → 768 | Expands the vector to `hidden_dim` and produces the values that get multiplied by the gate | 3rd (together with `w1`) |

**Confusing bit, explicitly flagged:** `w1`/`w2`/`w3` are numbered
arbitrarily — the numbers don't reflect execution order. `w1` and `w3` run
first (in parallel, both reading the same input), and `w2` runs **last**,
consuming the combined output of `w1` and `w3`. If you're skimming code
expecting `w1 → w2 → w3` in that order, you will misread it.

## Per-layer norms

`attention_norm` and `ffn_norm` are both RMSNorm layers. RMSNorm rescales a
vector's magnitude back to a consistent range — it exists because, without
it, values passing through many stacked layers tend to drift and grow (or
shrink) until the numbers become unstable. There's one of these before the
attention block and one before the feed-forward block, in every layer.

**Where `eps` comes from:** RMSNorm's epsilon (`1e-5`, guards against
dividing by zero when a vector's sum-of-squares is ~0) is **not** part of
the checkpoint's `Config` header — it's hardcoded as a constant in both
`run.c`'s `rmsnorm()` and `model.py`'s `ModelArgs.norm_eps` default. The
from-scratch engine (`main.cpp`) hardcodes the same `1e-5` for the same
reason: there's nothing in the `.bin` file to read it from.

## Outside the layers

Three tensors live outside the per-layer stack:

- **`tok_embeddings`** — shape 32000×288. A lookup table: row `i` is the
  vector representation of token ID `i`. Turning a token ID into a vector is
  just indexing into this array.
- **`norm`** — one final RMSNorm, applied once after all 6 layers, before
  the output projection.
- **`output`** — shape 288 → 32000. Projects the final vector back up to
  one score per vocabulary entry (the logits).

**Note:** `output` and `tok_embeddings` share the same underlying memory —
this was verified directly with `data_ptr()` (see the commented-out check in
`oracle.py`), they are literally the same tensor used in two directions
("weight tying"). It isn't two separate 32000×288 matrices; it's one, reused.

## Execution order within one layer

```
attention_norm -> wq/wk/wv -> attention -> wo -> ffn_norm -> w1/w3 -> w2
```

## Other terms

- **token** — a chunk of text (often a subword, not a whole word) that the
  model reads or writes one at a time. Think of it as one element of a
  tokenized stream, analogous to one entry in a pre-parsed array of symbols.
- **token ID** — an integer index identifying a token, into a fixed
  vocabulary table. Just an array index / enum value.
- **embedding** — a dense vector that stands in for a token ID. Produced by
  indexing a lookup table (`tok_embeddings`) with the token ID — same idea
  as `float table[vocab_size][dim]; vec = table[token_id];`.
- **logits** — the model's raw, un-normalized output scores, one per
  vocabulary entry. Not probabilities yet — just numbers, higher meaning
  "more likely," until something (softmax, sampling) turns them into a
  distribution.
- **tensor** — a multi-dimensional array. In this project, functionally the
  same as a struct holding a pointer plus a shape (like an ndarray or your
  own `Matrix` class).
- **forward pass** — running the input through the whole model once to
  produce an output. This is *the* computation — everything else (encoding,
  sampling, printing) is bookkeeping around it.
- **inference** — using an already-trained model to generate output, as
  opposed to training it (adjusting its weights). Everything this project
  does is inference; no training happens in `run.c`.
- **temperature** — a knob controlling how random the next-token choice is.
  0 = always pick the highest-scoring token (deterministic/greedy). Higher
  values flatten the distribution, making lower-scoring tokens more likely
  to get picked.
- **top-p** — an alternative sampling rule ("nucleus sampling"): only
  consider the smallest set of top-scoring tokens whose probabilities add up
  to at least `p`, and sample among just those.
- **seed** — the usual RNG seed. Fixes the random number sequence used for
  sampling, so a run is reproducible.
- **RMSNorm** — see "Per-layer norms" above: rescales a vector's magnitude
  without centering it (no mean subtraction, unlike the more common
  LayerNorm). Cheaper to compute, used throughout this model.
- **SwiGLU** — the specific feed-forward design used here: instead of a
  plain linear→activation→linear MLP, one branch (`w1`) computes a "gate"
  through the SiLU activation function, which is multiplied element-wise
  against a second branch (`w3`), before being projected back down (`w2`).
- **RoPE** (Rotary Position Embedding) — how the model encodes *where* in
  the sequence a token sits. Instead of adding a separate "position vector,"
  it rotates pairs of dimensions within the query/key vectors by an angle
  that depends on position. Applied inside `attention`, between `wq`/`wk`
  and the actual attention computation.
- **KV cache** — a store of previously-computed key/value vectors (`wk`,
  `wv` outputs) for tokens already processed, reused when generating each
  new token so they don't get recomputed from scratch every step. *(not
  needed yet — relevant once the from-scratch engine has a generation loop,
  not just a single forward pass)*
- **quantization** — representing weights (and sometimes activations) with
  fewer bits than 32-bit float, e.g. int8, to save memory and bandwidth at
  the cost of precision. This is what `runq.c` does. *(not needed yet —
  relevant once studying `runq.c` / the optimization phase)*
- **SIMD** — Single Instruction, Multiple Data: CPU instructions that apply
  one operation to several numbers at once (e.g. 8 floats in one
  instruction), instead of one at a time. Central to fast matmul. *(not
  needed yet — relevant to the optimization phase)*
- **matmul** — matrix multiplication. Important nuance for this project:
  it's always matrix **× vector** here, not matrix × matrix — the model
  processes one token at a time, so every matmul is a weight matrix times a
  single activation vector, not a batch of them.
- **forward hook** — a PyTorch mechanism that lets you attach a callback to
  a submodule, which fires with that submodule's output every time it runs
  during a forward pass. This is what `oracle.py` uses to capture every
  layer's activations without modifying `model.py`.
- **oracle** — a trusted reference implementation whose output you compare
  against, to check correctness of a new implementation. Here, the PyTorch
  model (`oracle.py`) is the oracle for the from-scratch C++ engine being
  built.
- **tolerance** — how much numerical difference between two implementations'
  outputs is acceptable before calling it a bug rather than floating-point
  noise. `compare.py` uses a 1e-4 relative tolerance.

## Model config: stories15M

```
dim          = 288
n_layers     = 6
n_heads      = 6
n_kv_heads   = 6
vocab_size   = 32000
hidden_dim   = 768
max_seq_len  = 256
norm_eps     = 1e-05
```

**Matmul count:** 43 matmul calls per token — 7 per layer (`wq`, `wk`, `wv`,
`wo`, `w1`, `w2`, `w3`) × 6 layers, plus 1 for the final `output` projection.
Confirmed by instrumentation, not just arithmetic.
