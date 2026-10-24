# GLM-5.3-Flash (`glm5next`)

GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`, Hugging Face `model_type` `glm5_next`) is a multimodal
mixture-of-experts model with 320B parameters, 18B of them active per token. jitllm loads its text
backbone from llama.cpp's GGUF format for it, architecture `glm5-next`, and runs it as the
`glm5next` family. The vision encoder is not supported yet.

llama.cpp's `convert_hf_to_gguf.py` converts the whole checkpoint. `scripts/glm5next/convert.py`
writes the same format and can convert a slice of the layers from only the shards that hold them;
apart from rounding in `ssm_a`, the two write identical tensors. `scripts/glm5next/reference.py` is
the PyTorch reference that the Java code is checked against.

## The checkpoint

All numbers below come from the official `config.json` and the safetensors headers.

| | |
|---|---|
| layers | 45 decoder layers, plus one MTP layer (`layers.45`) |
| hidden size | 4096, vocabulary 154 880 (the GLM-4 tokenizer and `glm4` pre-tokenizer) |
| token mixers | 34 KDA layers and 11 sparse-attention layers, in the order `lll d` repeated, then a final `l` |
| feed-forward | dense SwiGLU (12 288) in layers 0–2; in layers 3–44, 288 routed experts (2048), top 8, plus 1 shared expert |
| residual | 4 mHC streams around every sublayer, Sinkhorn with 20 iterations |
| positions | no RoPE anywhere; 1M context |
| storage | FP8 E4M3 with 128×128 block scales (`weight_scale_inv`) for the dense, expert, shared-expert and MLA `q_a`/`q_b`/`kv_a`/`o` projections; BF16 elsewhere; `A_log`, `dt_bias`, the mHC bias and scale, and the router bias are F32. The checkpoint is about 328 GB in 62 shards. |

### KDA (Kimi delta attention)

KDA has 64 heads of 128. Per token:

1. q, k and v are separate 4096→8192 projections. Each goes through its own depthwise causal
   convolution (width 4) and a SiLU.
2. q and k are L2-normalized per head as `x / sqrt(Σx² + 1e-6)`, and q is scaled by 128^-½.
3. The decay is per channel, not per head: `g = −5 · sigmoid(exp(A_log[h]) · (f_b(f_a(x)) + dt_bias))`,
   using a rank-128 `f_a`/`f_b` pair. The factor −5 is `gate_lower_bound`.
4. `beta = sigmoid(b_proj x)`, one value per head.
5. The state update is `S = diag(exp g) S`, then `S += k ⊗ (v − Sᵀk) β`, then `o = Sᵀq`.
6. The output is `o_norm(o)` (an RMS norm per head) times `sigmoid(g_b(g_a(x)))`, followed by `o_proj`.

### Sparse attention (DSA with k-pools)

The sparse-attention layers use MLA with no RoPE: 64 heads, q_lora 1536, kv_lora 512, key and
value head width 256, scale 256^-½. Each position caches only its normalized 512-wide latent.

Each layer has its own indexer (all `indexer_types` are `full`):

1. `q = wq_b(q_latent)` gives 32 heads of 128.
2. Each position stores a key `LayerNorm(wk x)` (with bias, eps 1e-6) and a 128-wide pool gate.
3. Each complete pool of 4 positions gets a compressed key. Per channel, it is
   `softmax_j(gate_j + ape[j]) · key_j`.
4. A pool's score is `Σ_h w_h/√32 · relu(q_h·key/√128)`, where `w = weights_proj x`.
5. The query attends to the 512 best complete pools (`index_topk` 2048 / 4) plus the incomplete
   pool that holds its own position. Up to 2051 positions are attended, so attention is dense
   until then.

### mHC (manifold-constrained hyper-connections)

Every sublayer works on 4 residual streams:

1. The flattened streams (16 384 values) are RMS-normalized without a weight.
2. `fn` (24×16 384) projects them to 24 values. With `base` and `scale`, these give:
   - `pre = sigmoid(·) + 1e-6`
   - `post = 2 · sigmoid(·)`
   - a 4×4 mix: a row softmax plus 1e-6, then Sinkhorn normalization.
3. The sublayer's input is `Σ pre_m · stream_m`.
4. The output streams are `post_k · out + Σ_m comb[m][k] · stream_m`.
5. The embedding is copied into all 4 streams.
6. The head takes the streams' mean, then `norm`, then `lm_head`.

### Feed-forward

- Both the dense and the expert SwiGLU clamp their inputs: `gate ≤ 10` and `up ∈ [−10, 10]`.
- Routing is DeepSeek-V3 style:
  - the top 8 of `sigmoid(logit) + e_score_correction_bias`;
  - each chosen expert is weighted by its sigmoid;
  - the weights are normalized and scaled by 2.5.

### MTP

The MTP layer is a plain pre-norm block without mHC. Its token mixer is sparse attention and its
feed-forward is the MoE.

- Input: `eh_proj([enorm(embed(next token)), hnorm(trunk hidden)])`, where the trunk hidden is the
  final normalized state.
- Output: `shared_head.norm`, then the shared `lm_head`.
- The embedding is zeroed at position 0, as vLLM does. transformers does not implement this layer.

## How it maps onto jitllm

| Block | Reused from | New |
|---|---|---|
| KDA | the qwen35 delta-rule layout: row = key channel, `CpuOperations.causalConv1d` with the GGUF channel-major taps, `Qwen35DeltaNetKernels` structure | per-channel decay, three separate convolutions, the FLA L2 norm (eps under the root), sigmoid output gate (`Glm5NextForward.kda`, `Glm5NextKernels.kda*`) |
| sparse attention | the deepseek2 absorbed MLA: `attn_k_b`/`attn_v_b` split as llama.cpp does, `DeepSeek2Kernels.absorbQuery` and `decompressLatent` with rope width 0 | latent-only cache (512, no rope tail), the k-pool indexer and selection (`indexerSelect`, `Glm5NextKernels.indexer*`), attention restricted to the selection (`sparseLatentAttention`) |
| MoE | the deepseek2 sigmoid routing with selection bias; the qwen35moe expert kernels | the SwiGLU clamp, 288 experts (beyond the 256 the existing router kernels hold), each expert tensor loaded as stacks that stay under 2³¹ bytes (`Glm5NextModelLoader`) |
| mHC | none | `hyperConnectionPre/Post`, `Glm5NextKernels.hc*` |
| MTP | the qwen35 MTP hook | `Glm5NextForward.forwardNextn` |
| tokenizer, chat | `Glm4Tokenizer` (same split regex, `ignore_merges`) and `Glm4ChatFormat` | GLM-5.3's template also opens with `<|system|>Reasoning Effort: …` and has the assistant start with `<think>` |

## Running on the GPU

The CUDA plan (`Glm5NextPlanProvider`) runs the trunk in all three modes: one token at a time,
sequential prefill/decode, and batched prefill/decode. Its requirements and limits:

- **Weights:** Q8_0 projections, with norms, gates, the router, the mHC parameters and the
  indexer's head weights and k-pool gate in F32 (what llama.cpp's converter and `convert.py` write
  with `q8_0`). Q4_0 and K-quant weights are converted to Q8_0 at load, so they take Q8_0's
  memory.
- **Expert stacks:** the routed experts must load as at most two stacks. In Q8_0 the full model's
  288 experts load as two stacks of 144.
- **Cache:** an FP32 latent cache, so run with `--fp32-kv-cache`.
- **Native libraries:** `--with-native-libraries` is refused; the kernels below are TornadoVM's.
  `--cuda-graphs` works in every mode.
- **Not supported yet:** splitting the model across devices, and the MTP layer.

### Decode

The whole trunk is one task graph (`Glm5NextLayers`, `Glm5NextKernels`). Per layer:

1. **mHC.** `rmsNormQuantize` normalizes and quantizes in one workgroup per 256 values, every
   workgroup taking the whole sum of squares in the same order. `hcMixes` splits each of the 24 mix
   sums over 16 chunks of the streams. `hcCoefficients` adds the chunks up and runs the Sinkhorn
   normalization in one workgroup, a lane per mix and per entry of the 4×4 matrix. `hcCollapse`
   and `hcExpand` are a lane per element.
2. **KDA.** The projections are the Q8_0 DP4A matrix-vector kernels, then qwen35's convolution,
   `kdaL2Norm`, `kdaDecayAndBeta` and `kdaGatedNorm`. `kdaDeltaRule` gives a workgroup 32 state
   columns of one head, each column's 128 rows split into 8 chunks, a lane per (chunk, column),
   the chunks' sums added in chunk order; the state is written once per token.
3. **Sparse attention.** deepseek2's `twoProjectionsQ8_0`, `latentNorms`, `ropeAndCache` (rotation
   width 0), `absorbQuery` and `decompressLatent`. The indexer's head weights and k-pool gate,
   which llama.cpp keeps in F32, go through `projectF32`. `indexerScores` gives each pool 32
   lanes, a head per lane, the heads' terms added in head order; `indexerSelect` and
   `sparseLatentAttention` follow.
4. **MoE.** `routerTopKSigmoid` (up to 512 experts), `expertsGateUpClamped`, and `expertsDown`,
   which reads only the stack that holds each slot's expert.

### Batched prefill

`Glm5NextBatchPrefillLayers` and `Glm5NextBatchKernels` run the trunk over a chunk of tokens in one
task graph, a row per token:

- **Projections:** int8 tensor-core GEMMs (`Int8GemmKernels.gemmInt8Q8_0`) over activations
  quantized per 32-value block, wherever the shape is whole GEMM tiles; DP4A kernels otherwise, and
  `projectF32` for the F32 indexer weights. The chunk's arrays are padded to whole 128-row tiles.
- **Routed experts:** the chunk's (token, expert) pairs are grouped by expert (`groupByExpert`) and
  multiplied as grouped int8 GEMMs (`groupedGateUpClamped`, `groupedDown`), then combined.
- **mHC:** `hcMixes` is one tiled GEMM, streams × `fn`ᵀ, split into 16 chunks of the depth
  (32 rows × 32 mixes per workgroup, staged in local memory); `hcFinishMixes` adds the chunks.
- **KDA:** the convolution walks the chunk's rows in order. `kdaDeltaRule` uses the decode
  kernel's layout, and keeps the workgroup's 128×32 slice of the state in local memory for the
  whole chunk, staging each row's key, query and decay once.
- **Sparse attention:** on FP16 tensor cores when the latent and value widths are whole 128-column
  tiles (`Glm5NextConfiguration.tensorCoreAttention`): the absorbed queries, their scores against
  every filled position of the cache, a softmax that keeps only the indexer's pools and the open
  tail (`sparseSoftmax`), the values and the decompression are strided GEMMs
  (`BatchMmaKernels.gemmMMAStrided`) with FP32 accumulation. Otherwise an online softmax over tiles
  of positions.
- **Indexer:** `indexerScores` gives each (row, pool) 32 lanes, as the decode kernel does.
- **Hand-off to decode:** the latent cache, the KDA windows and matrices, and the indexer caches
  are the decode layers' own arrays. The decode graph takes them, and every weight the prefill
  graph reads, from the prefill graph by name; `attn_k_b` and `attn_v_b`, which the tensor-core
  attention reads as FP16 copies, it binds itself.

### Agreement with the host

`Glm5NextDecodeParityAccelTest` compares the plan with the host pass over the tiny and medium
models' Q8_0 files. To make the comparison like for like, the host rounds the activation of every
projection to int8 per 32-value block, as the device kernels do.

- The single-layer models agree to float rounding (about 1e-7) at a quarter of the positions or
  more; the rest are about 1% off where an activation sits on an int8 rounding boundary.
- The whole tiny model, whose routers have near ties, is held to an error of 0.05 from the first
  token and to the same greedy choices.
- After a batched prefill, the medium model must track the host as closely as token-by-token
  decoding does. The prefill and the decode add some sums in different orders (the mHC mixes, the
  delta rule's chunks), so a rounding flip in the prefilled positions can stay in the cache.

## Performance

On an NVIDIA A100 80GB, with Q8_0 slices of the real weights in the llama.cpp format, jitllm with
`--cuda-graphs --fp32-kv-cache` and llama.cpp (October 2026, `-fa on -ngl 99`) on the same file.
Tokens per second; llama.cpp's first repetition, always slower, is left out of its numbers.

Layers 3–10 (8 MoE layers: 6 KDA, 2 sparse attention; 60 GiB):

| | jitllm | llama.cpp |
|---|---|---|
| prefill 512, batch 512 | 3566 | 2305 |
| prefill 1024, batch 512 | 3546 | 2317 |
| prefill 512, batch 256 | 2531 | 1481 |
| prefill 1024, batch 256 | 2551 | 1488 |
| decode 128 | 209 | 188 |

Layers 0–3 (3 dense layers and 1 MoE layer; 9.5 GiB):

| | jitllm | llama.cpp |
|---|---|---|
| prefill 512, batch 512 | 10385 | 8700 |
| prefill 1024, batch 512 | 10156 | 8795 |
| prefill 512, batch 256 | 8104 | 6432 |
| prefill 1024, batch 256 | 7991 | 6174 |
| decode 128 | 396 | 405 |

A slice says little about the whole model's speed: its embedding and output head weigh as much as
in the full model, against 4 or 8 of 45 layers.

## GGUF layout

The layout is llama.cpp's `glm5-next` (`src/models/glm5-next.cpp`); the key prefix is
`glm5-next.`. What the loader reads:

- `block_count`, which counts the MTP layers too, and `nextn_predict_layers`. A file can hold a
  slice of the layers, renumbered from 0.
- `attention.head_count_kv` per block: 0 for a KDA block, 1 for sparse attention.
  `leading_dense_block_count` blocks have a dense feed-forward, the rest a mixture of experts.
- `kda.head_dim`, `kda.gate_lower_bound`, `ssm.conv_kernel`; `attention.q_lora_rank`,
  `attention.kv_lora_rank`, `attention.key_length_mla`, `attention.value_length_mla`;
  `attention.indexer.{head_count,key_length,top_k,kpool,kpool_select_tail}`;
  `hyper_connection.{count,sinkhorn_iterations,epsilon}`; `expert_*` and `swiglu_clamp_exp`.

Tensors:

- mHC: `hc_{attn,ffn}_{fn,base,scale}`
- KDA: `attn_{q,k,v}`, `ssm_conv1d_{q,k,v}`, `ssm_f_{a,b}`, `ssm_dt.bias`, `ssm_a` (which holds
  `-exp(A_log)`), `ssm_beta`, `ssm_g_{a,b}`, `ssm_norm`, `attn_output`
- sparse attention: `attn_q_a`, `attn_q_a_norm`, `attn_q_b`, `attn_kv_a_mqa`, `attn_kv_a_norm`,
  `attn_k_b`, `attn_v_b`, `attn_output`
- indexer: `indexer.attn_q_b`, `indexer.attn_k`, `indexer.k_norm.{weight,bias}`,
  `indexer.proj`, `indexer_compressor_ape`, `indexer_compressor_gate`
- experts: `ffn_{gate,up,down}_exps`, one tensor per projection over every expert
- MTP: `nextn.{eh_proj,enorm,hnorm,shared_head_norm}`

Norms, biases, the router, the mHC parameters, the KDA convolutions, `ssm_a`, `ssm_dt` and the
indexer's head weights, k-pool gate and APE stay F32. Other matrices take the requested type; with
Q4_0, `convert.py` keeps the embedding and output in Q8_0.

## Checking

- `reference.py` builds a tiny model with every layer kind and random weights, written in the
  official checkpoint format with FP8, BF16 and F32 tensors. It records each layer's streams, the
  indexer selections, the logits and the MTP logits over 40 tokens. With `--check-transformers` it
  also runs the same checkpoint through the transformers implementation.
- `Glm5NextForwardTest` runs the converted F32 and Q8_0 files against those goldens.
  `Glm5NextKernelsAccelTest` checks the decode kernels against host evaluations, and
  `Glm5NextDecodeParityAccelTest` checks the whole decode plan as described above.
