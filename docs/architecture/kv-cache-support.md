# Key/value cache precision: support matrix

The key/value cache is stored in **FP16 by default**; accumulation is FP32 in every path.
FP32 (`--fp32-kv-cache`, `StorageOptions.fp32()`, `-Djitllm.kvcache.fp32=true`) is the
compatibility and numerical-reference choice and is supported everywhere.

A configuration is **supported** for FP16 only when the same path writes and reads the FP16
cache end to end and a numerical comparison against FP32 backs it. Accepting the setting is not
evidence: before this matrix existed, many paths allocated the FP16 arrays and then wrote and read
FP32 ones while reporting FP16, and Q8_0 batched prefill handed decode an FP16 cache that nothing
had written. Unsupported configurations are refused by `Fp16KeyValueSupport` before any cache or
plan is built — at model load, when a session's policy override selects another mode, and for
internal callers when a plan is built — with `GPUL-CFG-002`, the combination and the FP32 setting.
They never fall back to FP32 silently.

The decision keys on the **loaded** weight representation: a Q4_1, Q4_K, Q5_K or Q6_K file that is
materialized as Q8_0 runs, and is decided as, Q8_0.

## Evidence classes

- **tested**: numerical comparison against FP32 (`KvPrecisionHarness`: teacher-forced, per-row
  cosine, relative L2 and top-1 agreement, and a check that FP16 storage changed the result at all)
  on the named fixture, over a prompt that crosses prefill-chunk boundaries and a multi-step decode.
- **structural**: the same kernels and bindings as a tested row, differing only in a dimension the
  kernels do not see (for example model size within a family).
- **blocked**: no hardware or no model file here to test it; refused until it can be.

## Matrix

| Backend | Family | Weights (as loaded) | Mode | FP16 | Evidence |
| --- | --- | --- | --- | --- | --- |
| CPU | all families | all | all | supported | tested: Llama 1B Q8_0 (single-token and 32-token chunked prefill), Qwen3 0.6B F16, Qwen2.5 0.5B Q8_0, Granite 3.2 2B Q8_0, Phi-3 mini Q8_0 — `CpuFp16KvPrecisionAccelTest`: cosine ≥ 0.999998, relative L2 ≤ 0.0022, top-1 100% |
| CUDA | Llama, Qwen3 | F16 | single-token (Llama: lowered by default) | supported | tested: Llama 1B F16 cos ≥ 0.9999997, rel L2 ≤ 7.6e-4; Qwen3 0.6B F16 cos ≥ 0.9999977, rel L2 ≤ 0.0023; top-1 100% — `GpuFp16KvLlamaF16AccelTest`, `GpuFp16KvQwen3F16AccelTest` |
| CUDA | Llama, Qwen3 | F16 | batched prefill, tensor-core MMA (128-token chunks) | supported | tested: Llama 1B F16 cos ≥ 0.9999994, rel L2 ≤ 0.0011; Qwen3 0.6B F16 cos ≥ 0.9999974, rel L2 ≤ 0.0024; top-1 100%. Measured with the per-query flash attention; the tensor-core GEMM attention that replaced it gives identical batched greedy text on these models, and has not been re-run through the harness |
| CUDA | Llama | Q8_0 (and materialized Q8_0) | single-token, sequential prefill/decode, batched prefill (MMA) | supported | tested: Llama 1B Q8_0 — single-token and prefill/decode cos ≥ 0.9999996, rel L2 ≤ 9.3e-4; batched cos ≥ 0.9999997, rel L2 ≤ 7.5e-4; top-1 100% — `GpuFp16KvLlamaQ8AccelTest`. Structural: other Llama sizes (3B, 8B) and Q4_1/Q4_K/Q5_K/Q6_K files loaded as Q8_0 |
| CUDA | Llama | Q4_0 (retained) | single-token | supported | tested: Llama 1B Q4_0 cos ≥ 0.9999996, rel L2 ≤ 9.4e-4 — `GpuFp16KvLlamaQ4AccelTest` |
| CUDA | Qwen3.5 / Qwen3.8 | Q4_0 (+ mixed) | all | supported | `Qwen35Fp16Kv*` sequence-reset and parity goldens |
| CUDA | Qwen3 | Q8_0 (and materialized Q8_0) | single-token, sequential prefill/decode, batched prefill (MMA) | supported | tested: Qwen3 0.6B Q8_0 — single-token and prefill/decode cos ≥ 0.9999994, rel L2 ≤ 0.0012; batched cos ≥ 0.9999989, rel L2 ≤ 0.0015; top-1 100% — `GpuFp16KvQwen3Q8AccelTest` |
| CUDA | Llama, Qwen3 at 8B | Q8_0 | batched prefill (MMA) | supported | tested: Llama-3.1-8B Q8_0 cos ≥ 0.9999996, rel L2 ≤ 9.3e-4; Qwen3-8B Q8_0 cos ≥ 0.9999962, rel L2 ≤ 0.0028; top-1 100% — `GpuFp16KvLlama8BQ8AccelTest`, `GpuFp16KvQwen3_8BQ8AccelTest`; end-to-end generation coherent with `--cuda-graphs` (the reported configuration) |
| CUDA | Llama, Qwen3 | F16 | sequential prefill/decode | supported | tested: Llama 1B F16 cos ≥ 0.9999997, rel L2 ≤ 7.6e-4; Qwen3 0.6B F16 cos ≥ 0.9999977, rel L2 ≤ 0.0023; top-1 100% (the layers used to force the FP32 cache) |
| CUDA | Llama, Qwen3 | F16, Q8_0 | batched prefill without tensor cores (scalar kernels) | supported | structural: the scalar FP16 twins (`batchedRopeWithKVCacheFP16Paged`, `batchedRopeWithKVCacheQwen3FP16Paged`, `batchedFlashAttentionKVFP16Paged`) are the ones tested on OpenCL below; no pre-sm_80 CUDA device here |

| CUDA | Mistral | F16, Q8_0 | single-token | supported | tested: Mistral-7B Q8_0 cos ≥ 0.9999999, rel L2 ≤ 4.4e-4, top-1 100% — `GpuFp16KvMistralQ8AccelTest`. F16 structural (same cache kernels; projections do not touch the cache) and end-to-end identical text to FP32; its FP32/FP16 comparison needs two 14.5 GB plans in one JVM, which this 24 GB device cannot hold |
| CUDA | Qwen2 / DeepSeek-R1-Distill, Phi-3, Granite (3.2 and 4.0) | F16, Q8_0 | single-token | supported | tested: Qwen2.5 0.5B F16/Q8_0 cos ≥ 0.9999963, rel L2 ≤ 0.0029; DeepSeek-R1-Distill-Qwen 1.5B Q8_0 cos ≥ 0.9999971; Phi-3 mini F16/Q8_0 cos ≥ 0.9999994, rel L2 ≤ 0.0012; Granite 3.2 2B F16/Q8_0 cos ≥ 0.9999993, rel L2 ≤ 0.0013; Granite 4.0 1B F16/Q8_0 cos ≥ 0.9999997; top-1 100% — `GpuFp16Kv{Qwen2,DeepSeekQwen2,Phi3,Granite,Granite4}*AccelTest` |
| CUDA | Gemma 4 | Q8_0, Q4_0 (retained, mixed) | single-token, batched prefill | supported | tested: gemma-4-E2B-it Q8_0 single-token cos ≥ 0.99999999, rel L2 ≤ 1.0e-4, batched rel L2 ≤ 3.6e-4; Q4_0 single-token rel L2 ≤ 2.2e-4, batched ≤ 3.6e-4 with the packed-integer decode off (with it on, the eight-bit activation rounding amplifies the difference to 0.027); top-1 100% — `GpuFp16KvGemma4{Q8,Q4}AccelTest`. The packed path over the FP16 cache: long-prefix NLL screen (1100 positions past the 512-position window) within 0.002 nats/token of the host on both files |
| CUDA | Gemma 4 | F16 / BF16 | single-token, batched prefill | supported | structural: the F16 layers write the cache through the same FP16 RoPE and attention kernels as the Q8_0 rows. No numerical comparison against the FP32 cache yet |
| CUDA | Devstral | Q8_0, Q4_K_M, F16 | single-token, batched prefill (tensor cores) | supported | structural: Llama-shaped layers with the FP16 twins of the cache kernels; end-to-end batched greedy text identical to single-token on Devstral-Small-2-24B Q8_0 and Q4_K_M (A100). No `KvPrecisionHarness` run yet |
| CUDA | Qwen2-MoE | Q8_0 | single-token, batched prefill (tensor cores) | supported | structural: attention and cache kernels shared with Qwen2; end-to-end batched greedy text identical to single-token on Qwen1.5-MoE-A2.7B Q8_0 (A100). No `KvPrecisionHarness` run yet |
| CUDA | Qwen2 / DeepSeek-R1-Distill, Phi-3, Granite 3.2, Mistral | F16, Q8_0 | batched prefill (tensor cores) | supported | structural: the Llama tensor-core layers and their attention over the FP16 cache, tested above; end-to-end batched greedy text identical to single-token on Qwen2.5 0.5B Q8_0/F16, DeepSeek-R1-Distill-Qwen 1.5B, Granite 3.2 2B, Mistral-7B Q8_0 (A100); Phi-3 mini matches until split-K's summation order flips a near-tied token |
| OpenCL (NVIDIA-class device) | Llama, Qwen3, Qwen2 / DeepSeek, Phi-3, Granite 3.2 / 4.0, Mistral | F16, Q8_0, Q4_0 (Llama) | every mode each family has; batched prefill runs the scalar kernels | supported | tested on RTX 5090 Laptop via NVIDIA OpenCL, TornadoVM 7.0.1-dev `4c6b5819` — 23 rows, cos ≥ 0.9999952, rel L2 ≤ 0.0031, top-1 100% (same `GpuFp16Kv*AccelTest` classes under `--backend opencl`) |
| OpenCL (other devices) | any | any | any | refused (blocked) | non-NVIDIA scheduler layers keep an FP32 cache; no such device here |
| OpenCL (NVIDIA-class device) | Gemma 4 | Q8_0, Q4_0 (retained, mixed) | single-token | supported | tested on RTX 5090 Laptop via NVIDIA OpenCL, TornadoVM 7.0.1: gemma-4-E2B-it Q8_0 cos ≥ 0.99999999, rel L2 ≤ 1.4e-4; Q4_0 cos ≥ 0.99999999, rel L2 ≤ 2.6e-4; top-1 100% — `GpuFp16KvGemma4{Q8,Q4}AccelTest` under an OpenCL SDK. Decode attention is `attentionDecodeGroupFP16Shared`, the shared-memory twin of the CUDA shuffle kernel (OpenCL cannot compile `simdShuffleDown`): rel L2 ≤ 3.0e-6 against FP64 on the host — `Gemma4DecodeGroupSharedParityAccelTest`. Greedy text identical to CUDA's on the validation prompts, both caches |
| OpenCL (NVIDIA-class device) | Qwen3.5 / Qwen3.8 | Q4_0 (+ mixed) | single-token, sequential prefill/decode | supported | tested on RTX 5090 Laptop via NVIDIA OpenCL, TornadoVM 7.0.1: Qwen3.8-27B Q4_0 `Qwen35Fp16KvSequenceReset{Standard,PrefillDecode}AccelTest` (the per-head `processHeadsFlashAttentionFP16Paged`; split-KV stays CUDA's), greedy text identical to CUDA's on the validation prompts, both caches |
| OpenCL | Gemma 4, Qwen3.5 / Qwen3.8 | any | batched prefill | refused, either cache (`BatchPrefillSupport`) | gemma4's batched prefill is tensor-core only; qwen35's scalar batched projections do not compile on TornadoVM 7.0.1's OpenCL backend (`LogicConstantNode` in `OCLNodeLIRBuilder.emitIf`, a TornadoVM defect) |
| Metal | Llama, Qwen3 | F16, Q8_0 | single-token, sequential prefill/decode, batched prefill (SIMD-group) | supported | tested on an Apple M4 Pro, TornadoVM 7.2.1-dev (SIMD-group half fragments): Llama 1B F16 cos ≥ 0.9999987, rel L2 ≤ 0.0016; Llama 1B Q8_0 cos ≥ 0.9999994, rel L2 ≤ 0.0010; Qwen3 0.6B F16 cos ≥ 0.9999982, rel L2 ≤ 0.0020; Qwen3 0.6B Q8_0 cos ≥ 0.9999993, rel L2 ≤ 0.0012; top-1 100% — `GpuFp16Kv{Llama,Qwen3}{F16,Q8}AccelTest` |
| Metal | Llama | Q4_0 (retained) | single-token, batched prefill | supported | tested on M4 Pro: Llama 1B Q4_0 cos ≥ 0.9999997, rel L2 ≤ 8.8e-4, top-1 100% — `GpuFp16KvLlamaQ4AccelTest` |
| Metal | Qwen2 / DeepSeek-R1-Distill, Phi-3, Granite 3.2 / 4.0, Mistral | F16, Q8_0 | single-token, batched prefill | supported | tested on M4 Pro: Qwen2.5 1.5B F16/Q8_0 cos ≥ 0.9999990, rel L2 ≤ 0.0015; DeepSeek-R1-Distill-Qwen 1.5B Q8_0 cos ≥ 0.9999975, rel L2 ≤ 0.0023; Phi-3 mini F16/Q8_0 cos ≥ 0.9999993, rel L2 ≤ 0.0013; Granite 3.2 2B F16/Q8_0 cos ≥ 0.9999981, rel L2 ≤ 0.0019; Granite 4.0 1B Q8_0 cos ≥ 0.9999998, rel L2 ≤ 7.8e-4; Mistral-7B Q8_0 cos ≥ 0.9999999, rel L2 ≤ 3.5e-4; top-1 100%. F16 of Granite 4.0 and Mistral structural (no file here) |
| Metal | Qwen3.5 | Q4_0 (+ mixed), Q8_0 | all | supported | tested on M4 Pro: Qwen3.5-0.8B Q4_0 cos ≥ 0.9999997, rel L2 ≤ 7.2e-4, top-1 100% — `GpuFp16KvQwen35Q4AccelTest`; Qwen3.5-0.8B Q8_0 cos ≥ 0.99999993, rel L2 ≤ 3.9e-4, top-1 100% — `GpuFp16KvQwen35Q8AccelTest`; greedy text identical to the FP32 cache over a 20- and a 1379-token prompt. F16 stays refused (no file here) |
| Metal | Gemma 4 | Q8_0 | single-token, batched prefill | supported | tested on M4 Pro: gemma-4-E2B-it Q8_0 single-token cos ≥ 0.99999999, rel L2 ≤ 1.1e-4; batched rel L2 ≤ 3.1e-4; top-1 100% — `GpuFp16KvGemma4Q8AccelTest` |
| Metal | Gemma 4 Q4_0; Qwen2 / DeepSeek, Phi-3, Granite, Mistral in sequential prefill/decode; every other family | any | — | refused (blocked) | no model file here (Gemma 4 Q4_0), or the path was not measured on Metal |
| CUDA (tensor-core) | continuous batching (`serve --continuous-batching`, experimental): Llama, Qwen3 | F16 | batched decode over the shared pool | supported | tested, token streams only (the engine samples on the device): FP16 and FP32 pools give identical greedy streams — Llama 1B F16 41 and 28 tokens to the stop token, Qwen3 0.6B F16 64 and 64 tokens, two concurrent requests with 150- and 300-token prompts — `EngineFp16KvAccelTest`; the decode kernels are FP16-storage twins of the FP32 ones |

## Metal

What the FP16 cache runs on Metal, and what it costs. Apple GPUs allow 32 KB of threadgroup
memory, so the 64-thread per-key FP16 split-KV kernel (34052 bytes) cannot be built there:

- **Decode attention**: the lane-cooperative split-KV kernel
  (`processHeadsFlashAttentionSplitKVFP16PagedLaneHead128`) for 128-wide heads (Qwen3), the
  per-head flash kernel (`processHeadsFlashAttentionFP16Paged`) for Llama, Mistral, Qwen2,
  Phi-3 and Granite, the lane-cooperative split-KV kernels for Qwen3.5's 256-wide heads
  (`processHeadsFlashAttentionSplitKV{,FP16}PagedLaneHead256`, either cache), and Gemma 4's
  shuffle-reduced grouped kernel (`attentionDecodeGroupFP16`, rel L2 ≤ 4.2e-7 against FP64 —
  `Gemma4DecodeGroupSharedParityAccelTest`).
- **Batched prefill attention**: `TransformerPagedKvBatchPrefillSimdgroupFP16Kernels` (64-, 96-
  and 128-wide heads, plain and Granite-scaled) and `Gemma4BatchPrefillSimdgroupAttentionFP16Kernels`
  (256 and 512): the FP32 SIMD-group kernels with key and value fragments loaded as
  `simdgroup_half8x8` and multiplied with `float` query and probability fragments
  (`*SimdgroupFP16KernelsAccelTest`, host reference with NaN in unwritten cache and padding
  rows). Qwen3.5 runs `attentionBatchFP16Paged`; its score-scratch kernel is slower on Metal.
- **Greedy text**: identical to the FP32 cache, single-token and batched (chunk 256), on all 16
  model files below over a 400-token prompt and 64 generated tokens, with one exception:
  Granite 4.0 1B Q8_0 single-token flips one near-tied token, and the CPU path with an FP16
  cache flips the same token, so it is the cache's rounding rather than a Metal kernel.

Measured on an Apple M4 Pro, `jitllm bench --gpu --with-prefill-decode --batch-prefill-size 256
--pp 512,2048 --tg 128`, tok/s, FP16 cache / FP32 cache:

| Model | pp512 | pp2048 | tg128 |
| --- | --- | --- | --- |
| Qwen3-0.6B F16 | 4197 / 4126 | 3098 / 2870 | 157.9 / 154.7 |
| Qwen3-0.6B Q8_0 | 3846 / 3790 | 2910 / 2694 | 200.5 / 194.2 |
| Llama-3.2-1B F16 | 2331 / 2329 | 2147 / 2091 | 84.3 / 66.4 |
| Llama-3.2-1B Q8_0 | 2045 / 2081 | 1931 / 1936 | 114.0 / 86.2 |
| Llama-3.2-1B Q4_0 | 2099 / 2101 | 1953 / 1923 | 116.3 / 86.8 |
| Qwen2.5-1.5B F16 | 1678 / 1674 | 1539 / 1505 | 51.2 / 51.9 |
| Qwen2.5-1.5B Q8_0 | 1520 / 1513 | 1402 / 1378 | 59.7 / 59.1 |
| DeepSeek-R1-Distill-Qwen-1.5B Q8_0 | 1539 / 1522 | 1407 / 1379 | 59.3 / 58.3 |
| Phi-3-mini F16 | 609 / 628 | 571 / 554 | 23.9 / 24.0 |
| Phi-3-mini Q8_0 | 572 / 559 | 521 / 503 | 30.4 / 30.3 |
| Granite-3.2-2B F16 | 936 / 928 | 858 / 853 | 36.8 / 36.6 |
| Granite-3.2-2B Q8_0 | 840 / 837 | 781 / 775 | 52.1 / 50.9 |
| Granite-4.0-1B Q8_0 | 1341 / 1334 | 1158 / 1121 | 52.4 / 52.2 |
| Qwen3.5-0.8B Q4_0 | 2097 / 2021 | 1459 / 1150 | 101.1 / 99.6 |
| Gemma-4-E2B Q8_0 | 1076 / 1063 | 992 / 975 | 67.2 / 68.3 |
| Mistral-7B Q8_0 | 296 / 287 | 279 / 270 | 18.1 / 15.1 |

The Llama-shaped decode gains most (Llama 1B +27% to +34%, Mistral +20%): their FP32 cache
on Metal runs the per-key `processHeadsParallelPaged`, the FP16 cache the flash kernel.
Everywhere else the FP16 cache is within noise in decode and equal or faster in prefill, faster
at 2048 positions, where half the cache bytes are read.
