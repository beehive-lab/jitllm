# GLM-5.3-Flash (`glm5next`)

GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`, Hugging Face `model_type` `glm5_next`) is a multimodal
mixture-of-experts model with 320B parameters, 18B of them active per token. jitllm loads its text
backbone under the GGUF architecture name `glm5next`. The vision encoder is not supported yet.

No llama.cpp GGUF exists for this model. `scripts/glm5next/convert.py` converts the official
safetensors checkpoint, and `scripts/glm5next/reference.py` is the PyTorch reference that the
Java code is checked against.

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
| MoE | the deepseek2 sigmoid routing with selection bias; the qwen35moe expert kernels | the SwiGLU clamp, 288 experts (beyond the 256 the existing router kernels hold), expert stacks split into tensors of at most `expert_chunk_size` experts so each stays under 2³¹ elements |
| mHC | none | `hyperConnectionPre/Post`, `Glm5NextKernels.hc*` |
| MTP | the qwen35 MTP hook | `Glm5NextForward.forwardNextn` |
| tokenizer, chat | `Glm4Tokenizer` (same split regex, `ignore_merges`) and `Glm4ChatFormat` | GLM-5.3's template also opens with `<|system|>Reasoning Effort: …` and has the assistant start with `<think>` |

## Running on the GPU

The CUDA decode plan (`Glm5NextPlanProvider`, `Glm5NextLayers`) runs the trunk one token at a
time, in the standard and sequential prefill/decode modes. It has these requirements and limits:

- **Weights:** Q8_0 projections, with norms, gates, the router and the mHC parameters in F32 (what
  `convert.py --type Q8_0` writes).
- **Expert stacks:** the routed experts must be stored in at most two stacks. The full model's 288
  experts in stacks of 144 qualify.
- **Cache:** an FP32 latent cache, so run with `--fp32-kv-cache`.
- **Not supported yet:** batched prefill, splitting the model across devices, and the MTP layer.

The whole trunk is one task graph. Per layer:

1. The mHC kernels (`hcMixes`, `hcCoefficients`, `hcCollapse`, `hcExpand`) wrap each sublayer.
2. KDA uses the Q8_0 DP4A projections, qwen35's convolution, and `kdaL2Norm`, `kdaDecayAndBeta`,
   `kdaDeltaRule` and `kdaGatedNorm`.
3. Sparse attention uses deepseek2's `twoProjectionsQ8_0`, `latentNorms`, `ropeAndCache` (rotation
   width 0), `absorbQuery` and `decompressLatent`, plus the indexer kernels and
   `sparseLatentAttention`.
4. The MoE uses `routerTopKSigmoid` (up to 512 experts), `expertsGateUpClamped` and `expertsDown`
   over the two stacks.

`Glm5NextDecodeParityAccelTest` compares the plan with the host pass over the tiny model's Q8_0
files. To make the comparison like for like, the host rounds the activation of every projection to
int8 per 32-value block, as the device kernels do. Agreement:

- At most positions the two agree to float rounding (about 1e-7).
- The other positions are about 1% off. An activation sat on an int8 rounding boundary, and the two
  paths' last-bit differences rounded it apart. Nudging the host's own input by one ulp reproduces
  the same positions.

## GGUF layout written by the converter

The architecture key prefix is `glm5next.`. Main keys:

- `block_count`, plus `layer_index`, `attention_types` (0 KDA, 1 sparse) and `ffn_types`
  (0 dense, 1 MoE) per layer. A file can hold a slice of the layers, renumbered from 0.
- `nextn_predict_layers`
- `kda.*`, `indexer.*`, `hc.*`, `expert_*` (including `expert_chunk_size`), `swiglu_limit`.

Tensor names follow llama.cpp where it has one. The new tensors are:

- mHC: `hc_{attn,ffn}_{fn,base,scale}`
- KDA: `kda_conv_{q,k,v}`, `kda_f_{a,b}`, `kda_dt_bias`, `kda_a_log`, `kda_beta`, `kda_g_{a,b}`,
  `kda_o_norm`
- indexer: `idx_q_b`, `idx_k`, `idx_k_norm.{weight,bias}`, `idx_weights`, `idx_pool_ape`,
  `idx_pool_gate`
- experts: `ffn_{gate,up,down}_exps.<chunk>`
- MTP: `nextn.{eh_proj,enorm,hnorm,shared_head_norm}`

Norms, biases, gates, the router and the mHC parameters stay F32. Other matrices take the
requested type; with Q4_0, the embedding and output stay Q8_0.

## Checking

- `reference.py` builds a tiny model with every layer kind and random weights, written in the
  official checkpoint format with FP8, BF16 and F32 tensors. It records each layer's streams, the
  indexer selections, the logits and the MTP logits over 40 tokens. With `--check-transformers` it
  also runs the same checkpoint through the transformers implementation.
- `Glm5NextForwardTest` runs the converted F32 and Q8_0 files against those goldens.
  `Glm5NextKernelsAccelTest` checks the decode kernels against host evaluations, and
  `Glm5NextDecodeParityAccelTest` checks the whole decode plan as described above.
