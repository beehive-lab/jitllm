# DeepSeek-V4 (`deepseek4`)

DeepSeek-V4 (`deepseek-ai/DeepSeek-V4-Flash-Base`, Hugging Face `model_type` `deepseek_v4`,
architecture `DeepseekV4ForCausalLM`) is a mixture-of-experts model with about 292B parameters
(including the MTP layer), about 13B of them active per token. jitllm loads it from llama.cpp's GGUF
format for it, architecture `deepseek4` (`src/models/deepseek4.cpp`), and runs it as the
`deepseek4` family.

This page describes the model, how it maps onto jitllm, the GGUF layout and the GPU plan. Most of
its blocks come from two families jitllm already has. Its mHC, indexer and expert loading are the
`glm5next` family's (GLM-5.3-Flash, see [glm5next.md](glm5next.md)). Its multi-head latent
attention is related to `deepseek2`'s.

## The checkpoint

All numbers below come from the official `config.json` and the safetensors headers.

| | |
|---|---|
| layers | 43 decoder layers, plus one MTP layer (`mtp.0`) |
| hidden size | 4096, vocabulary 129 280 (the DeepSeek-V3 tokenizer), untied output head |
| attention | 64 query heads of 512 (64 of them rotary), one shared key/value head; q_lora 1024; a sliding window of 128 in every layer |
| compression | `compress_ratios`: 0 in layers 0–1, then 4 and 128 alternating (4 in even layers 2–42, 128 in odd layers 3–41); the 44th entry, 0, is the MTP layer's |
| output projection | 8 groups, each 8 heads × 512 → 1024 (`wo_a`), then 8192 → 4096 (`wo_b`) |
| feed-forward | every layer is MoE: 256 routed experts (2048), top 6, plus 1 shared expert; layers 0–2 are hash-routed |
| residual | 4 mHC streams around every sublayer, Sinkhorn with 20 iterations, and a learned mHC head |
| positions | RoPE on the last 64 dimensions: base 10 000 without scaling in layers 0–1; base 160 000 (`compress_rope_theta`) with YaRN ×16 (original 65 536) in the compressed layers; 1M context |
| storage | FP8 E4M3 with 128×128 block scales (`.scale`, F32 holding powers of two) for `wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`, the indexer's `wq_b`, the shared expert and the routed experts; BF16 for the norms, the compressors, the indexer's `weights_proj`, the router, the embedding and the head; F32 for the sinks, the APEs, the mHC parameters and the router bias; I64 for the hash table. The checkpoint is 294.7 GB in 46 shards. |

The checkpoint uses DeepSeek's own tensor names (`layers.N.attn.wq_a.weight`, `embed.weight`,
`head.weight`, `mtp.0.*`), not the `model.layers.*` names transformers uses. Each layer sits in its
own shard: shard 1 holds `embed`, shards 2–44 hold layers 0–42 (6.6 GB each), shard 45 holds `head`,
`norm` and `hc_head_*`, and shard 46 holds the MTP layer (8.7 GB, with its own copies of the
embedding and head). A slice of layers needs only shards 1, 45 and that slice's own shards.

### Other checkpoints

| | layers | hidden | heads | experts | routed experts stored as | size |
|---|---|---|---|---|---|---|
| V4-Flash-Base | 43 | 4096 | 64 | 256 × 2048, top 6 | FP8, 128×128 blocks | 294.7 GB |
| V4-Flash (instruct) | 43 | 4096 | 64 | 256 × 2048, top 6 | FP4 E2M1 with an E8M0 scale per 32 values | 159.6 GB |
| V4-Pro-Base / V4-Pro | 61 | 7168 | 128 | 384 × 3072, top 6 | FP8 / FP4 | 1606 GB (Base) |

V4-Flash differs from Flash-Base only in the routed experts. Their FP4 layout is MXFP4's, so
llama.cpp repacks them into `MXFP4` blocks without loss. V4-Pro also has `q_lora_rank` 1536,
`o_groups` 16, `index_topk` 1024 and routed scale 2.5, and its first two layers have ratio 128.
DeepSeek-V4.1 (`DeepseekV41ForCausalLM`) is a different architecture, with Engram embeddings,
cross-layer KV and indexer sharing, and a vision tower. This family does not cover it.

### Attention

Every layer runs the same query and key/value path. For a token at position `p`:

1. `qr = q_norm(wq_a x)`, then `q = wq_b qr`, as 64 heads of 512. Each head is RMS-normalized
   without a weight (eps 1e-6).
2. `kv = kv_norm(wkv x)`, a single 512-wide vector. It is both the key and the value.
3. RoPE rotates the last 64 dimensions of each query head and of `kv`. Rotation is on interleaved
   pairs `(2i, 2i+1)`, as in ggml's normal RoPE mode. In the compressed layers, llama.cpp sets its
   attention factor so that YaRN changes only the frequencies, not the magnitude, as the reference
   does.
4. Attention uses scale 512^-½ and a learned sink per head. The softmax denominator gets an extra
   `exp(sink_h)` term, as in gpt-oss.
5. The output's last 64 dimensions are rotated back (by `−p`).
6. The 64 heads form 8 groups of 8 heads (4096 values). Group `g` is projected to 1024 values by its
   own slice of `wo_a`. The 8 results are concatenated and projected by `wo_b` (8192 → 4096).

The keys a query sees depend on the layer's compression ratio:

- **Ratio 0 (layers 0–1, and the MTP layer):** the last 128 positions, including its own. The
  window cache is a ring of 128 vectors of 512.
- **Ratio 128 ("HCA", 20 layers):** the window, plus every complete compressed block before it. A
  block holds 128 positions, so at 1M tokens a query sees 128 + 8192 keys.
- **Ratio 4 ("CSA", 21 layers):** the window, plus the 512 compressed blocks of 4 positions that the
  indexer scores highest (all of them while there are 512 or fewer).

A compressed block becomes visible once its last position has been processed. Window keys and
compressed keys sit in one score row, sharing one softmax with the sink.

### The compressor

A compressor turns each block of `r` positions into one 512-wide key by gated pooling:

1. Every position computes `kv_j = wkv_c x_j` and `score_j = wgate_c x_j + ape[j mod r]`, in FP32.
2. When a block completes, its key is `Σ_j softmax_j(score_j) · kv_j`, per channel.
3. The pooled key is RMS-normalized (`compressor.norm`), and its last 64 dimensions are rotated
   with the compressed RoPE at the block's first position, `p + 1 − r`.

For ratio 128, `wkv_c` and `wgate_c` are 4096 → 512 and `ape` is 128 × 512.

For ratio 4 the windows overlap. `wkv_c` and `wgate_c` are 4096 → 1024 and `ape` is 4 × 1024. The
first 512 channels of a position feed the *next* block, and the last 512 feed its own block. A block's
softmax runs over 8 entries per channel: the previous block's 4 positions (first halves) and its
own 4 (second halves). Block 0 has no previous block, so those entries are padded with −∞ scores.
Decoding keeps a state of 8 rows of 1024 values and 1024 scores per layer, and shifts it when a block
completes.

### The indexer

Each ratio-4 layer has a lightning indexer. It is a second, smaller compressor plus a scorer:

1. `q_I = wq_b_I qr`, as 64 heads of 128, with the compressed RoPE on the last 64 dimensions.
2. Its compressor works like the attention compressor, at width 128 (`wkv`/`wgate` 4096 → 256,
   `ape` 4 × 256, overlapping). It caches one 128-wide key per block.
3. `w = weights_proj x · (128 · 64)^-½`.
4. A block's score is `Σ_h w_h · relu(q_I,h · k_block)`.
5. The top 512 complete blocks are selected.

The reference applies a Hadamard rotation to `q_I` and to the compressed indexer keys. A rotation
applied to both sides does not change the dot products. It only matters because the reference then
rounds them to FP4, to match its quantization-aware training (see below).

### mHC

The hyper-connections are GLM-5.3's: 4 streams, `fn` of 24 × 16 384, `pre = sigmoid + eps`,
`post = 2·sigmoid`, a row softmax plus eps, one column normalization, then 19 rounds of row and
column normalization. llama.cpp names the fields `hc_{attn,ffn}_{fn,base,scale}` in both models.
The only difference is the head:

- GLM-5.3 averages the 4 streams.
- DeepSeek-V4 weights them, with `output_hc_fn` (4 × 16 384), `output_hc_base` and
  `output_hc_scale`: `pre = sigmoid(scale · fn·rmsnorm(streams) + base) + eps`, then `Σ pre_m · stream_m`.

### Feed-forward

- **Router scoring:** `s = sqrt(softplus(gate x))`, in FP32. llama.cpp refuses every other gating
  function for this architecture.
- **Score-routed layers (3–42, MTP):** pick the top 6 of `s + bias`.
- **Hash-routed layers (0–2):** take the 6 experts from `tid2eid[token]`, a table of 129 280 × 6.
  These layers have no bias.
- **Weights:** each chosen expert is weighted by its `s`. The weights are normalized to sum 1 and
  scaled by 1.5.
- **SwiGLU clamp:** both the routed experts and the shared expert clamp their inputs, `gate ≤ 10`
  and `up ∈ [−10, 10]`, as GLM-5.3 does.

### MTP

The MTP layer is a full layer with mHC, ratio 0 and score routing. Its input is
`e_proj(enorm(embed(next token))) + h_proj(hnorm(streams))`, applied to each of the 4 streams of the
last trunk layer. Its output goes through the mHC head, `norm` and the head. The checkpoint gives the
MTP layer its own `hc_head_*`. llama.cpp's graph uses the trunk's instead.

### Quantization-aware details

The official reference rounds some activations to match its quantization-aware training:

- the first 448 dimensions of every window and compressed key, to FP8 in blocks of 64 with
  power-of-two scales;
- the indexer's queries and keys, to FP4 in blocks of 32;
- every FP8 or FP4 matrix input, to FP8 in blocks of 128.

llama.cpp and transformers do none of this. jitllm follows them. `reference.py` writes its goldens
without the rounding, and can enable it to measure how large the gap is.

## How it maps onto jitllm

| Block | Reused from | New |
|---|---|---|
| mHC | `glm5next`: `hyperConnectionPre/Post`, `Glm5NextKernels.hc*`, `Glm5NextBatchKernels.hcMixes/hcFinishMixes` | the learned head (`output_hc_*`), a variant of `hcCollapse` with its own weights |
| query path | deepseek2's `rmsNormQuantize`, `twoProjectionsQ8_0`, `latentNorms` | per-head unweighted RMS norm of `q`; interleaved RoPE on the tail with YaRN (deepseek2's `ropeAndCache` rotates NEOX halves and caches a latent) |
| attention | the online-softmax and split structure of `DeepSeek2Kernels.attentionSplit`/`combineSplits`; `glm5next`'s `sparseLatentAttention` (attention over a selection) and `sparseSoftmax` | a single 512-wide key/value head attended directly, with no absorbed `k_b`/`v_b` decompression; sinks; window ring plus compressed blocks in one softmax; rotating the output back |
| output | the Q8_0 matrix-vector and int8 GEMM kernels | grouped `wo_a`: 8 independent 4096 → 1024 products, as one strided batch |
| compressor | `glm5next`'s k-pool gating (`indexerPoolKey(s)`: `softmax(gate + ape) · key` over a pool of 4) | pools of 128 and overlapping pools of 4 with two halves, per-channel gates 512/1024 wide, the pooled key's RMS norm and RoPE, the decode state buffers |
| indexer | `glm5next`'s `indexerScores` (`Σ w_h relu(q_h·k)`) and `indexerSelect` | its own overlapping compressor instead of `LayerNorm(wk x)`; RoPE on the indexer queries; scale (128·64)^-½ |
| MoE | `glm5next`: `routerTopKSigmoid`'s selection with bias, `expertsGateUpClamped`, `expertsDown`, `groupByExpert`, the grouped GEMMs, the expert stacks under 2³¹ bytes (`Glm5NextModelLoader`) | `sqrtsoftplus` scoring; the hash-routed path (`tid2eid` gather, no bias); clamping the shared expert |
| MTP | `glm5next`'s `forwardNextn` hook | `e_proj`/`h_proj` (llama.cpp concatenates them into `eh_proj`), applied to 4 streams |
| tokenizer | the byte-level BPE in `tokenizer/` | the DeepSeek-V3 split rules (digits in groups of three, CJK runs, then the main pattern); the Base model needs no chat format |

New classes, mirroring `glm5next`:

- model: `model/deepseek4/{DeepSeek4, DeepSeek4Configuration}`
- loading: `DeepSeek4ModelLoader`, `DeepSeek4Provider`
- state: `DeepSeek4State`, holding the window rings, the compressor states, the compressed and
  indexer caches, and the 4 streams
- weights: `DeepSeek4LayerWeights` with standard and tornado variants
- CPU: `backend/cpu/DeepSeek4Forward`
- kernels: `DeepSeek4Kernels`, `DeepSeek4BatchKernels`
- layers: `DeepSeek4Layers`, `DeepSeek4BatchPrefillLayers`
- plan: `DeepSeek4PlanProvider` and its components
- workspace: `DeepSeek4DeviceArrays`, `DeepSeek4BatchArrays`
- recognition: `GgufRecognition` maps `general.architecture = deepseek4` to the family, with a
  `ModelType` constant and the Allowlists entries for it.

## GGUF layout

The layout is llama.cpp's `deepseek4`; the key prefix is `deepseek4.`. What the loader reads:

- `block_count` and `nextn_predict_layers`. The trunk file has no MTP layer: llama.cpp's converter
  skips `mtp.*` unless asked for an MTP-only file, which holds just the MTP layer.
- `embedding_length` (4096), `embedding_length_out` (4 × 4096), `attention.head_count`,
  `attention.key_length`/`value_length` (512), `rope.dimension_count` (64), and
  `rope.freq_base` with the YaRN `rope.scaling.*` keys.
- `attention.q_lora_rank`, `attention.sliding_window`, `attention.compress_ratios` (one entry per
  block, at least `block_count` of them), `attention.compress_rope_freq_base`,
  `attention.output_group_count`, `attention.output_lora_rank`,
  `attention.layer_norm_rms_epsilon`.
- `attention.indexer.{head_count,key_length,top_k}`.
- `hyper_connection.{count,sinkhorn_iterations,epsilon}`.
- `expert_count`, `expert_used_count`, `expert_feed_forward_length`, `expert_shared_count`,
  `expert_weights_scale`, `expert_weights_norm`, `expert_gating_func` (4 = `sqrtsoftplus`),
  `swiglu_clamp_exp`, `swiglu_clamp_shexp` and `hash_layer_count`.

Tensors (llama.cpp's shapes, innermost dimension first):

- top level: `token_embd`, `output_norm`, `output`, `output_hc_{fn,base,scale}`
- mHC: `blk.N.hc_{attn,ffn}_{fn,base,scale}`
- attention: `attn_norm`, `attn_sinks`, `attn_q_a`, `attn_q_a_norm`, `attn_q_b`, `attn_kv`,
  `attn_kv_a_norm`, `attn_output_a` ({512·8, 1024·8}, read as {4096, 1024, 8}), `attn_output_b`
- compressor (ratio ≠ 0): `attn_compressor_{kv,gate,ape,norm}`
- indexer (ratio 4): `indexer.proj`, `indexer.attn_q_b`, `indexer_compressor_{kv,gate,ape,norm}`
- feed-forward: `ffn_norm`, `ffn_gate_inp`, `exp_probs_b.bias` (score-routed layers) or
  `ffn_gate_tid2eid` (I32 {6, 129 280}, hash-routed layers), `ffn_{gate,up,down}_exps` (one tensor
  per projection over all experts), `ffn_{gate,up,down}_shexp`
- MTP: `nextn.{eh_proj,enorm,hnorm,shared_head_norm}`, with optional `nextn.embed_tokens` and
  `nextn.shared_head_head`

Tensor types from llama.cpp's converter:

| Source | Written as |
|---|---|
| FP8 matrices | Q8_0, from the FP8 values times their block scales |
| BF16 matrices (compressors, `weights_proj`, router, embedding, head) | BF16 |
| F32 tensors (norms, sinks, APEs, mHC, bias) | F32 |
| FP4 routed experts | MXFP4, repacked without loss |
| hash table | I32 |

### The converters

**llama.cpp's converter.** It handles V4-Flash, whose experts are FP4. It cannot convert
V4-Flash-Base as it stands. `dequant_model` turns every FP8 weight, the routed experts included,
into floats and drops its `.scale`. The MXFP4 writer then requires a `.scale` beside every expert
weight and raises `KeyError`. (This comes from reading the converter; it has not been run.)

**`scripts/deepseek4/convert.py`.** It writes the same format, from only the shards a slice needs:

- **Experts:** FP8 experts become Q8_0, which is close to lossless, or MXFP4 if requested. FP4
  experts become MXFP4, repacked exactly as llama.cpp does.
- **Other weights:** as in the table above. With `--f32-small`, the BF16 compressor, indexer and
  router weights are also written as F32, the way `glm5next` keeps its indexer weights.
- **Slices:** a slice of layers `[a, b)` is renumbered from 0. It gets `compress_ratios[a:b]` (plus
  one 0 entry for an MTP layer) and `hash_layer_count = max(0, 3 − a)`.

### MXFP4 in jitllm

`GGMLType` has no MXFP4 (ggml type 39: blocks of 32, one E8M0 byte and 16 bytes of E2M1 codes).
It is needed to load llama.cpp's V4-Flash files, and it halves the experts' memory. Two steps:

1. **Load as Q8_0.** An MXFP4 block decodes to `k · 2^(e−128)` with `k` in
   {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12} (ggml's doubled codes). That is a Q8_0 block with
   `d = 2^(e−128)` and `q = k`. The conversion is exact whenever `d` is a normal FP16 value
   (2^−14 ≤ d ≤ 2^15). The loader checks this, and falls back to rounding outside that range. Every
   existing Q8_0 kernel then runs unchanged, at Q8_0's memory.
2. **Native MXFP4.** Expert kernels decode the codes through a 16-entry table into int8, and use the
   E8M0 scale as an exponent: DP4A for decode, int8 tensor cores for the grouped GEMMs. This makes
   the full Flash model's experts 3.4 GB per layer instead of 6.8 GB.

## Running on the GPU

The plan follows `glm5next`: one task graph for the trunk in decode, one for batched prefill, Q8_0
projections, F32 small tensors, an FP32 cache, and TornadoVM kernels only (no native libraries).

### Memory

| | per layer | 43 layers | embedding + head |
|---|---|---|---|
| Q8_0 experts | ≈ 7.0 GB (experts 6.84, attention and shared expert ≈ 0.16) | ≈ 301 GB | 1.06 GB BF16 embedding (host) + 0.56 GB Q8_0 head |
| MXFP4 experts | ≈ 3.6 GB | ≈ 155 GB | same |

- **Expert stacks:** one projection's Q8_0 stack is 256 × 2048 × 4096 values, 2.28 GB, just over
  2³¹ bytes. It loads as two stacks of 128, as `glm5next` does. MXFP4 stacks fit in one.
- **One A100 80GB:** about 9–10 layers with Q8_0 experts, or 18–20 with MXFP4 experts.
- **The full model:** at least 4 such GPUs with Q8_0 experts, or 2 with MXFP4. That needs the
  model split across devices, which comes after the single-device plan.
- **The test slice:** layers 0–3 of Flash-Base cover every layer kind: hash routing, ratio 0,
  ratio 4 with the indexer, and ratio 128. It needs shards 1–5 and 45 (about 30 GB), and makes a
  Q8_0 file of about 30 GB.
- **Caches:** the caches are small. With an FP32 cache, per token:
  - a ratio-4 layer stores 512 B of keys and 128 B of indexer keys;
  - a ratio-128 layer stores 16 B;
  - every layer also holds a fixed 128 × 512 window (256 KB);
  - at 1M tokens the whole trunk's cache is about 14 GB.

### Decode

Per layer, in one task graph:

1. **mHC pre:** `glm5next`'s `rmsNormQuantize`, `hcMixes`, `hcCoefficients` and `hcCollapse`.
2. **Query and key/value:**
   - `wq_a` and `wkv` as one two-projection Q8_0 kernel, then `latentNorms`;
   - `wq_b`;
   - a per-head norm and interleaved RoPE in one kernel;
   - `kv` written into the window ring.
3. **Compressor:** `wkv_c` and `wgate_c` (F32 or BF16, `projectF32`). One kernel updates the state.
   On a block boundary it also pools, normalizes, rotates and appends the block to the layer's
   compressed cache. It works on flat arithmetic blends, without branches that merge values.
4. **Indexer (ratio 4):**
   - `wq_b_I` with RoPE;
   - its compressor as in step 3;
   - `weights_proj`;
   - `indexerScores` over the complete blocks, then `indexerSelect` (top 512).
5. **Attention:** one kernel per (head group, split of keys). It runs an online softmax over the
   window entries, then over the selected blocks (ratio 4) or every complete block (ratio 128). A
   combine step adds the sink and rotates the output back. All 64 heads share the one key/value
   head. Each key is loaded once per workgroup and used by several heads, as deepseek2's absorbed
   attention does.
6. **Output:** grouped `wo_a` (one workgroup row per group and output, reading only its group's 4096
   inputs), then `wo_b`.
7. **mHC post,** then mHC pre again for the feed-forward.
8. **MoE:**
   - a router kernel with `sqrtsoftplus`; it selects through the bias, or reads `tid2eid` in hash
     layers;
   - the shared expert, as three Q8_0 projections with the clamp;
   - `expertsGateUpClamped` and `expertsDown` over the expert stacks;
   - the expert results and the shared expert added together.
9. **mHC post.**

After the last layer: the learned mHC head, `norm`, and the output projection.

Kernel rules that apply throughout:

- shapes travel in an `IntArray` when a kernel would take 18 or more arguments;
- values chosen by a condition are arithmetic blends, not merged after an if/else;
- indices the backend may compute speculatively (window slot, block index, expert id) are clamped.

### Batched prefill

The prefill graph processes a chunk of tokens, a row per token, as `glm5next`'s does:

- **Projections:** int8 tensor-core GEMMs (`Int8GemmKernels.gemmInt8Q8_0`), with the grouped
  `wo_a` as 8 strided GEMMs.
- **Routed experts:** grouped GEMMs after `groupByExpert`.
- **Compressor:** pools every block the chunk completes in one pass. The state carried over from
  before the chunk serves as the "previous block" of the first one.
- **Indexer:** scores each (row, block) pair, with causal visibility per row.
- **Attention:** each row sees the window (the 128 positions up to its own, from the cache and the
  chunk), plus its selected or visible compressed blocks. Attention runs on FP16 tensor cores
  when the shapes are whole tiles: 512-wide keys are whole 128-column tiles. The softmax keeps only
  the window and the selected blocks.
- **Hand-off to decode:** the window rings, compressor states and caches are the decode graph's own
  arrays, handed over by name.

## Checking

1. **`scripts/deepseek4/reference.py`.** A PyTorch implementation that follows the official
   `inference/model.py`, without the quantization-aware rounding unless asked.
   - It builds a tiny model with every layer kind: hash and score routing, ratios 0, 4 and 128, the
     indexer, the MTP layer, and a window of 8.
   - The model is written in the official checkpoint format (FP8 with scales, BF16, F32, I64; FP4
     experts optionally).
   - It records each layer's streams, the compressed keys, the indexer selections, the logits and
     the MTP logits over 300 tokens, enough for two ratio-128 blocks.
   - With `--check-transformers`, it also runs the checkpoint through transformers'
     `DeepseekV4ForCausalLM`, where the installed transformers provides it.
2. **`DeepSeek4ForwardTest`.** Runs the converted F32 and Q8_0 files on the CPU against those
   goldens.
3. **GPU tests.** `DeepSeek4KernelsAccelTest` checks the kernels against host evaluations, and
   `DeepSeek4DecodeParityAccelTest` checks the whole plan, decode and batched prefill.
4. **Real weights.** The layers 0–3 slice is checked against llama.cpp on the same file. That needs
   `llama-cli` or `llama-eval-callback`, not just `llama-bench`.
5. **Benchmarks.** `llama-bench` on the same GGUF.
