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
| CUDA | Llama, Qwen3 | F16 | single-token (legacy and lowered) | supported | pre-existing FP16 kernels; to be backed by a GPU harness row |
| CUDA | Llama, Qwen3 | F16 | batched prefill, tensor-core MMA | supported | pre-existing FP16 kernels; to be backed by a GPU harness row |
| CUDA | Qwen3.5 / Qwen3.8 | Q4_0 (+ mixed) | all | supported | `Qwen35Fp16Kv*` sequence-reset and parity goldens |
| CUDA | Llama, Qwen3 | Q8_0 (and materialized Q8_0) | all | refused | Q8_0 layers keep an FP32 cache; batched prefill hand-off relayed an unwritten FP16 cache |
| CUDA | Llama | Q4_0 (retained) | single-token | refused | Q4_0 layers keep an FP32 cache |
| CUDA | Llama, Qwen3 | F16 | sequential prefill/decode | refused | layers force the FP32 cache |
| CUDA | Llama, Qwen3 | F16 | batched prefill without tensor cores | refused | scalar batched-prefill kernels write FP32 only |
| CUDA | Llama, Qwen3 | any | non-NVIDIA scheduler | refused | non-NVIDIA decode layers keep an FP32 cache |
| CUDA | Mistral, Devstral, Qwen2 / DeepSeek-R1-Distill, Phi-3, Granite, Qwen2-MoE, Gemma 4 | any | any | refused | layers keep an FP32 cache |
| OpenCL | any | any | any | refused | no verified FP16 cache path |
| Metal | any | any | any | refused (blocked) | no verified FP16 cache path; no Metal device here |
| CUDA | parallel serving (`serve --parallel N`, N > 1) | F16 | batched decode | refused | engine kernels read an FP32 pool; `--fp32-kv-cache` required |
