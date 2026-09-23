# Key/value cache precision: support matrix

The key/value cache is stored in **FP16 by default**; accumulation is FP32 in every path.
FP32 (`--fp32-kv-cache`, `StorageOptions.fp32()`, `-Djllm.kvcache.fp32=true`) is the
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
| CUDA | Llama, Qwen3 | F16 | batched prefill, tensor-core MMA (128-token chunks) | supported | tested: Llama 1B F16 cos ≥ 0.9999994, rel L2 ≤ 0.0011; Qwen3 0.6B F16 cos ≥ 0.9999974, rel L2 ≤ 0.0024; top-1 100% |
| CUDA | Llama | Q8_0 (and materialized Q8_0) | single-token, sequential prefill/decode, batched prefill (MMA) | supported | tested: Llama 1B Q8_0 — single-token and prefill/decode cos ≥ 0.9999996, rel L2 ≤ 9.3e-4; batched cos ≥ 0.9999997, rel L2 ≤ 7.5e-4; top-1 100% — `GpuFp16KvLlamaQ8AccelTest`. Structural: other Llama sizes (3B, 8B) and Q4_1/Q4_K/Q5_K/Q6_K files loaded as Q8_0 |
| CUDA | Llama | Q4_0 (retained) | single-token | supported | tested: Llama 1B Q4_0 cos ≥ 0.9999996, rel L2 ≤ 9.4e-4 — `GpuFp16KvLlamaQ4AccelTest` |
| CUDA | Qwen3.5 / Qwen3.8 | Q4_0 (+ mixed) | all | supported | `Qwen35Fp16Kv*` sequence-reset and parity goldens |
| CUDA | Qwen3 | Q8_0 (and materialized Q8_0) | single-token, sequential prefill/decode, batched prefill (MMA) | supported | tested: Qwen3 0.6B Q8_0 — single-token and prefill/decode cos ≥ 0.9999994, rel L2 ≤ 0.0012; batched cos ≥ 0.9999989, rel L2 ≤ 0.0015; top-1 100% — `GpuFp16KvQwen3Q8AccelTest` |
| CUDA | Llama, Qwen3 at 8B | Q8_0 | batched prefill (MMA) | supported | tested: Llama-3.1-8B Q8_0 cos ≥ 0.9999996, rel L2 ≤ 9.3e-4; Qwen3-8B Q8_0 cos ≥ 0.9999962, rel L2 ≤ 0.0028; top-1 100% — `GpuFp16KvLlama8BQ8AccelTest`, `GpuFp16KvQwen3_8BQ8AccelTest`; end-to-end generation coherent with `--cuda-graphs` (the reported configuration) |
| CUDA | Llama, Qwen3 | F16 | sequential prefill/decode | supported | tested: Llama 1B F16 cos ≥ 0.9999997, rel L2 ≤ 7.6e-4; Qwen3 0.6B F16 cos ≥ 0.9999977, rel L2 ≤ 0.0023; top-1 100% (the layers used to force the FP32 cache) |
| CUDA | Llama, Qwen3 | F16, Q8_0 | batched prefill without tensor cores (scalar kernels) | supported | structural: the scalar FP16 twins (`batchedRopeWithKVCacheFP16Paged`, `batchedRopeWithKVCacheQwen3FP16Paged`, `batchedFlashAttentionKVFP16Paged`) are the ones tested on OpenCL below; no pre-sm_80 CUDA device here |

| CUDA | Mistral | F16, Q8_0 | single-token | supported | tested: Mistral-7B Q8_0 cos ≥ 0.9999999, rel L2 ≤ 4.4e-4, top-1 100% — `GpuFp16KvMistralQ8AccelTest`. F16 structural (same cache kernels; projections do not touch the cache) and end-to-end identical text to FP32; its FP32/FP16 comparison needs two 14.5 GB plans in one JVM, which this 24 GB device cannot hold |
| CUDA | Qwen2 / DeepSeek-R1-Distill, Phi-3, Granite (3.2 and 4.0) | F16, Q8_0 | single-token | supported | tested: Qwen2.5 0.5B F16/Q8_0 cos ≥ 0.9999963, rel L2 ≤ 0.0029; DeepSeek-R1-Distill-Qwen 1.5B Q8_0 cos ≥ 0.9999971; Phi-3 mini F16/Q8_0 cos ≥ 0.9999994, rel L2 ≤ 0.0012; Granite 3.2 2B F16/Q8_0 cos ≥ 0.9999993, rel L2 ≤ 0.0013; Granite 4.0 1B F16/Q8_0 cos ≥ 0.9999997; top-1 100% — `GpuFp16Kv{Qwen2,DeepSeekQwen2,Phi3,Granite,Granite4}*AccelTest` |
| CUDA | Devstral, Qwen2-MoE, Gemma 4 | any | any | refused (blocked) | layers keep an FP32 cache; no model files here to implement and test against |
| OpenCL (NVIDIA-class device) | Llama, Qwen3, Qwen2 / DeepSeek, Phi-3, Granite 3.2 / 4.0, Mistral | F16, Q8_0, Q4_0 (Llama) | every mode each family has; batched prefill runs the scalar kernels | supported | tested on RTX 5090 Laptop via NVIDIA OpenCL, TornadoVM 7.0.1-dev `4c6b5819` — 23 rows, cos ≥ 0.9999952, rel L2 ≤ 0.0031, top-1 100% (same `GpuFp16Kv*AccelTest` classes under `--backend opencl`) |
| OpenCL (other devices) | any | any | any | refused (blocked) | non-NVIDIA scheduler layers keep an FP32 cache; no such device here |
| OpenCL | Qwen3.5 / Qwen3.8 | any | any | refused | verified on CUDA only |
| Metal | any | any | any | refused (blocked) | no verified FP16 cache path; no Metal device here |
| CUDA | continuous batching (`serve --continuous-batching`, experimental) | F16 | batched decode | refused | engine kernels read an FP32 pool; `--fp32-kv-cache` required |
