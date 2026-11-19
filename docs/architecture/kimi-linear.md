# Kimi-Linear (`kimi-linear`)

Kimi-Linear-48B-A3B (`moonshotai/Kimi-Linear-48B-A3B-Instruct`, Hugging Face `model_type`
`kimi_linear`) is a mixture-of-experts model with 48B parameters, about 3B of them active per
token. jitllm loads it from llama.cpp's GGUF format for it, architecture `kimi-linear`, and runs it
as the `kimi-linear` family. llama.cpp's own implementation (`src/models/kimi-linear.cpp`) is the
reference the Java code is checked against, on the same files.

## The checkpoint

All numbers below come from the official `config.json` and `modeling_kimi.py`.

| | |
|---|---|
| layers | 27 decoder layers, no MTP layer |
| hidden size | 2304, vocabulary 163 840 (Kimi's tiktoken BPE) |
| token mixers | 20 KDA layers and 7 MLA layers: MLA at layers 3, 7, 11, 15, 19, 23 and 26 (0-based), KDA elsewhere, about 3 to 1 |
| feed-forward | dense SwiGLU (9216) in layer 0; in layers 1–26, 256 routed experts (1024), top 8, plus 1 shared expert (1024) |
| residual | a plain pre-norm residual (no hyper-connections), RMS norm with eps 1e-5 |
| positions | no RoPE anywhere; 1M context (`model_max_length` 1 048 576) |
| storage | BF16; `A_log` and `dt_bias` F32 |

The config's `kda_layers` and `full_attn_layers` count layers from 1.

### KDA (Kimi delta attention)

KDA has 32 heads of 128 (`linear_attn_config.head_dim`). Per token:

1. q, k and v are separate 2304→4096 projections. Each goes through its own depthwise causal
   convolution (width 4) and a SiLU.
2. q and k are L2-normalized per head as `x / sqrt(Σx² + eps)`, and q is scaled by 128^-½.
   llama.cpp uses the model's RMS epsilon (1e-5) there; FLA uses 1e-6.
3. The decay is per channel: `g = −exp(A_log[h]) · softplus(f_b(f_a(x)) + dt_bias)` (FLA's
   `fused_kda_gate`), with a rank-128 `f_a`/`f_b` pair. llama.cpp's converter stores
   `−exp(A_log)` as `ssm_a`.
4. `beta = sigmoid(b_proj x)`, one value per head.
5. The state update is `S = diag(exp g) S`, then `S += k ⊗ (v − Sᵀk) β`, then `o = Sᵀq`.
6. The output is `o_norm(o)` (an RMS norm per head) times `sigmoid(g_b(g_a(x)))`, followed by `o_proj`.

### MLA (latent attention without RoPE)

32 heads; no compressed query (`q_lora_rank` null): `q_proj` is 2304→32·192. `kv_lora_rank` 512,
`qk_nope_head_dim` 128, `qk_rope_head_dim` 64, `v_head_dim` 128, scale 192^-½. `mla_use_nope`: the
64-wide "rope" part of the query and of the shared key is used as it is, never rotated. llama.cpp
runs it absorbed: each position caches its normalized 512-wide latent and its 64-wide key tail
(576 values), and a head's query is its no-rope part pushed through `attn_k_b` followed by its tail.

### Feed-forward

- No SwiGLU clamp.
- Routing is DeepSeek-V3 style with one group: the top 8 of `sigmoid(logit) + e_score_correction_bias`;
  each chosen expert is weighted by its sigmoid; the weights are normalized and scaled by 2.446.
- The shared expert is ungated and added to the routed experts.

### Tokenizer and chat template

- The tokenizer is tiktoken with Kimi's `pat_str` (Han runs on their own, letter runs that exclude
  Han and take an English contraction in any case, digits in threes). llama.cpp writes it as
  byte-level BPE with `tokenizer.ggml.pre = kimi-k2`; a chunk the vocabulary holds whole is one
  token, as tiktoken encodes it. The special tokens run from `[BOS]` (163584) to the end.
- Chat: `<|im_user|>user<|im_middle|>…<|im_end|>`, the same with `<|im_system|>` and
  `<|im_assistant|>`, no separator between turns, and the generation prompt
  `<|im_assistant|>assistant<|im_middle|>`. No token opens the conversation. Generation stops at
  `<|im_end|>` (163586) and `[EOS]` (163585).

## How it maps onto jitllm

Kimi-Linear is deepseek2's latent attention and mixture of experts plus glm5next's KDA. The pieces
the two KDA families share are family-neutral classes:

| Block | Reused from | New |
|---|---|---|
| KDA | glm5next: the host pass (`KdaForward`, `KdaHostState`, `KdaWeights`), the decode kernels (`KdaKernels`: L2 norm, delta rule, gated norm) and qwen35's convolution | the softplus decay (`KdaForward.Decay.SOFTPLUS`, `KdaKernels.kdaSoftplusDecayAndBeta`); glm5next's is `−5 · sigmoid(…)` |
| MLA | deepseek2: the absorbed form with a 576-wide cache (`DeepSeek2Kernels.ropeAndCache`, `absorbQuery`, `attentionSplit`, `combineSplits`, `decompressLatent`, and their FP16-cache forms) | the query straight from `attn_q` (no `q_a`/`q_b`); the key tail "rotated" by identity RoPE tables, so the deepseek2 kernels run unchanged |
| MoE | glm5next's routed-expert decode kernels, now `RoutedExpertKernels`, with the clamp at infinity; deepseek2's router logits | none |
| residual | the deepseek2 layout: one hidden state, residual-fused output projections | no hyper-connections, unlike glm5next |
| tokenizer, chat | `Qwen3Tokenizer` (byte-level BPE, merge priority = merged token id = tiktoken rank) | `KimiTokenizer` (Kimi's split, whole-chunk lookup, specials from `[BOS]`), `KimiChatFormat` |

Compared with glm5next's blocks:

- **KDA:** the same shape of computation; 32 heads instead of 64, `f_a`/`g_a` rank 128 in both. Only
  the decay differs (softplus instead of a bounded sigmoid).
- **MLA vs KPool-DSA:** Kimi's MLA is dense: no indexer, no k-pools, no selection. It has no
  compressed query and keeps a 64-wide unrotated key tail, where glm5next's sparse attention has
  `q_a`/`q_b`, no tail (width 0) and 256-wide heads.
- **Router/experts:** both are sigmoid routing with a selection bias and one shared expert; Kimi has
  256 experts of 1024 (one stack each, 0.6 GB in Q8_0), no clamp, scale 2.446.
- **No mHC:** one residual stream; nothing collapses or expands around the sublayers.

## Running on the GPU

The CUDA plan (`KimiLinearPlanProvider`) runs one token at a time, as a sequential prefill followed
by decode, and with batched prefill, on one device or split by layer across several (single-token
or batched). Every stage keeps the whole model's cache layout.

- **Weights:** Q8_0 projections; the routed experts all Q8_0 or all Q4_0 (dot products on the
  nibbles as stored, with DP4A). Norms, the router, its bias, the convolutions, `ssm_a` and
  `dt_bias` are F32, as llama.cpp's converter writes them.
- **Cache:** FP16 by default, FP32 with `--fp32-kv-cache`.
- **Decode** (`KimiLinearLayers`): a stage's layers in one task graph; KDA through `KdaKernels`,
  MLA through deepseek2's absorbed kernels, the experts through `RoutedExpertKernels`.
- **Batched prefill** (`KimiLinearBatchPrefillLayers`): int8 tensor-core GEMMs for the projections
  (`attn_kv_a_mqa` and `ssm_beta` padded to whole 128-column tiles), the KDA convolution and delta
  rule walking the chunk's rows in order (`KdaBatchKernels`), deepseek2's FP16 tensor-core latent
  attention, and grouped int8 GEMMs for the experts (`RoutedExpertBatchKernels`, Q4_0 recentred as
  its tiles are staged).
- **The seed token:** a new conversation is seeded with the prompt's own first token, since Kimi's
  prompts open with a role token rather than `[BOS]`; the prefill/decode loops would otherwise feed
  an extra `[BOS]` at position zero.

### Agreement with the host

On the CPU, Q8_0 greedy text is identical to llama.cpp's CPU output for 96 tokens. On the GPU, with
every run forced to the CPU's tokens, the worst relative logit error is about 5% for both
single-token decode and batched prefill, with FP16 or FP32 cache; the five most likely tokens agree
at every step and the top token differs only at near-ties.

## Performance

Tokens per second, against llama.cpp `11fe021` (`-fa 1 -ngl 99`) on the same file. Both sides on
the same 427-token prompt and a decode of 128 tokens at depth 427; jitllm medians with their range,
llama.cpp mean ± deviation.

NVIDIA A100 80GB, Q8_0 (48.65 GiB), five repetitions:

| | jitllm | llama.cpp |
|---|---|---|
| prefill, chunk / ubatch 256 | 2125 (2119–2130) | 1681 ± 88 |
| prefill, chunk / ubatch 512 | 2672 (2665–2685) | 2653 ± 106 |
| decode at depth 427 | 135–137 | 134 ± 0.7 |
| single-token plan, prompt and decode | 154 | — |

CUDA graphs change neither number.

2× NVIDIA A10, Q8_0 projections with Q4_0 routed experts (26.72 GiB), split by layer with the
default byte-balanced split (`--split-transport host`; llama.cpp `-sm layer`), three repetitions:

| | jitllm | llama.cpp |
|---|---|---|
| prefill, chunk / ubatch 256 | 1587 (1582–1590) | 1407 ± 13 |
| prefill, chunk / ubatch 512 | 2059 (2038–2074) | 1933 ± 52 |
| decode at depth 427 | 117 (106–118) | 112 ± 0.5 |
| single-token plan, prompt and decode | 127 | — |

## GGUF layout

The layout is llama.cpp's `kimi-linear`; the key prefix is `kimi-linear.`. What the loader reads:

- `block_count`; `attention.head_count_kv` per block: 0 for a KDA block, otherwise MLA;
  `leading_dense_block_count` blocks have a dense feed-forward, the rest a mixture of experts.
- `kda.head_dim`, `ssm.conv_kernel`; `attention.kv_lora_rank`, `attention.key_length_mla` (192),
  `attention.value_length_mla` (128), `rope.dimension_count` (64, the key tail);
  `expert_count`, `expert_used_count`, `expert_feed_forward_length`, `expert_shared_count`,
  `expert_weights_scale`, `expert_gating_func` (2, sigmoid).

Tensors:

- KDA: `attn_{q,k,v}`, `ssm_conv1d_{q,k,v}`, `ssm_f_{a,b}`, `ssm_dt.bias`, `ssm_a`, `ssm_beta`,
  `ssm_g_{a,b}`, `ssm_norm`, `attn_output`
- MLA: `attn_q`, `attn_kv_a_mqa`, `attn_kv_a_norm`, `attn_k_b`, `attn_v_b`, `attn_output`
- feed-forward: `ffn_{gate,up,down}` (layer 0); `ffn_gate_inp`, `exp_probs_b.bias`,
  `ffn_{gate,up,down}_exps`, `ffn_{gate,up,down}_shexp`

jitllm reads one file; a split GGUF (such as bartowski's Q8_0) must be merged with
`llama-gguf-split --merge` first. A file with Q8_0 projections and Q4_0 routed experts comes from a
Q8_0 file with `llama-quantize --allow-requantize --tensor-type ffn_gate_exps=q4_0 --tensor-type
ffn_up_exps=q4_0 --tensor-type ffn_down_exps=q4_0 in.gguf out.gguf Q8_0`.
