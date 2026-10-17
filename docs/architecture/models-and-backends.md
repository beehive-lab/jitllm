# Models and backends

How a model family is added, how weights become device data, and what each backend can and
cannot do.

## Adding a model family

A family is a set of `ServiceLoader` providers plus a program description. No central
`switch` is edited — the architecture rules forbid one (rule 15).

| Provider | Answers |
| --- | --- |
| `ModelProvider` | is this GGUF mine, and how do I load it? |
| `ModelArchitecture` | what is the layer topology, and what program does it describe? |
| `CpuForwardProvider` | how does it run on the CPU? |
| `TornadoPlanProvider` | how does it build task graphs? |
| `TornadoLoweringProvider` | how does it lower to the shared workspace plan? |
| `KvStorageFactory` | how is its KV storage laid out? |

Recognition maps a GGUF's declared `general.architecture` — `llama`, `qwen2`,
`granitemoehybrid`, `mistral3` — to an `ArchitectureId`, using `general.name` only where
the format leaves no other signal: Mistral and older Devstral builds ship as `llama`, and
a DeepSeek distill ships as `qwen2`, so the name is the only thing separating them. A
declared architecture that is not special-cased is passed through as its own identity, and
an identity nobody registered is reported as unrecognized (`GPUL-MOD-002`). It is never
mapped to the nearest family.

Registered architectures today: Llama, Mistral, Qwen2, Qwen3, DeepSeek-R1-Distill-Qwen,
Granite, Phi-3, Gemma-4, Devstral, Qwen3.5 (`qwen35`).

`qwen35` needed no recognition case at all — the file declares that architecture, nothing
else claims the name, and the pass-through branch resolved it. A provider class and one
service line were the whole registration, which is what rule 15 is for. It is also the first
family with **two kinds of layer**: only every fourth trunk layer attends, and the other
three mix with a Gated Delta Net recurrence holding fixed-size state instead of a key/value
cache.

It reuses Qwen 3's tokenizer, turn structure and thinking control, and **does not** reuse its
tool-call format: Qwen 3 puts a JSON object inside `<tool_call>`, and Qwen 3.5 puts nested
pseudo-XML there (`<function=name><parameter=x>`). Inheriting the wrong one would have produced a
model that converses correctly and gets tool calling silently wrong in both directions.

Gemma 4 has a third tool format, taken from the chat template in its GGUF files: declarations,
calls and results are written in a bare-key dictionary syntax whose strings are delimited by the
`<|"|>` token (`<|tool_call>call:getWeather{city:<|"|>Paris<|"|>}<tool_call|>`), and every marker
is a single token. The results go inside the assistant turn that made the calls, which stays open
for the answer, so the conversation encoder does not add a new assistant header after them. The
model ends a call with `<|tool_response>` or `<eos>`, both tool-aware stop tokens.

### Tool calling per family

Each family's tool rendering follows the chat template embedded in its GGUF. The unit tests
compare the encoder's output with that template rendered by Jinja2
(`src/test/resources/chat-templates/`). A family whose template has no tool format reports
`toolCalling = false`; the engine does not invent one.

| Family | Tools | Format (from the template) |
| --- | --- | --- |
| Llama 3.1 / 3.2 | yes | tools as `tojson(indent=4)` in the first user message; `{"name", "parameters"}` calls; `ipython` results |
| Qwen 2.5, Qwen 3 | yes | `<tools>` in the system turn; `<tool_call>{json}</tool_call>`; a run of results in one user turn |
| Qwen 3.5 (`qwen35`) | yes | as Qwen 3, but calls in `<function=…><parameter=…>` pseudo-XML and tools before the system text |
| Granite 3.2 | yes | a `tools` turn; `<\|tool_call\|>` + a JSON list; `tool` turns |
| Granite 4.0 | yes | `<tools>` in the system message; `<tool_call>` blocks; results in one user turn |
| Gemma 4 | yes | see above |
| DeepSeek-R1-Distill-Qwen | no | the template replays tool calls and outputs but never renders tool definitions |
| Qwen 1.5 / Qwen 2 MoE | no | no tools in the template, no `<tool_call>` in the vocabulary |
| Phi-3 mini | no | no tools in the template (GGUF or upstream) |
| Mistral v0.3, Devstral | no, for now | the Mistral v0.3 GGUF template has no tools; the upstream one does (`[AVAILABLE_TOOLS]`), but it needs this format's non-tool turns reworked first |

Granite's two dialects are told apart by the file's own template. A Granite file whose template
is neither dialect has no tool calling.

## Data types and materialization

| `DataType` | Quantized | Block-structured |
| --- | --- | --- |
| `F32`, `F16`, `BF16` | no | no |
| `Q8_0` | yes | no |
| `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K` | yes | yes |

**Dequantization is a materialization concern**, not a model concern and not a kernel
concern. The runtime tensor vocabulary names no file-format type; the format layer parses
GGUF and the loading path materializes weights in the representation the chosen backend
executes. Where a K-quant is decoded is therefore the backend's decision: the CPU backend
may keep a decoded representation, and the TornadoVM backend keeps Q4_K and Q6_K resident
on the device rather than expanding them on the host.

**Retaining a representation is a per-family capability, not a global one.** `Q4_K`/`Q6_K` are
retained for Devstral and `Q4_0` for Llama, because those families' layer graphs have kernels
that decode them in place; the same file loaded for a family without them is still materialized
as `Q8_0`. The loader opts in explicitly rather than retaining everything with a kernel
somewhere, because a tensor retained in a format its graph cannot read would be decoded as the
format it is not — 18-byte blocks addressed as 34-byte ones, which yields weights of plausible
magnitude and fluent, wrong text.

Retaining is worth roughly half a model's device footprint. Measured on one file,
`Llama-3.2-1B-Instruct-Q4_0.gguf`, switching only `-Djitllm.q4_0.retain`:

| | device peak | decode |
| --- | --- | --- |
| retained as `Q4_0` | 1570 MiB | 172.4 tok/s |
| materialized as `Q8_0` | 2060 MiB | 136.0 tok/s |

Faster as well as smaller, because single-token decode is bandwidth-bound: fewer bytes per
weight is fewer bytes read per token.

A backend declares which representations it accepts. A combination it does not accept is
refused, not silently converted — a silent conversion changes the arithmetic and shows up
as a numerical result nobody can attribute.

## Backends

`BackendId` is `CPU`, `CUDA`, `OPENCL` or `METAL`. CUDA, OpenCL and Metal are
*TornadoVM* backends: capabilities of one jitllm backend, selected by which SDK is
installed, not separate jitllm backends.

The launcher detects installed backends from `$TORNADOVM_HOME/etc/tornado.backend`. On a
multi-backend SDK, `--cuda`/`--opencl`/`--metal` force one and set TornadoVM's own
device-0 priority to match; on a single-backend SDK they are redundant but harmless. A
backend the SDK does not contain is an error, not a fallback.

## Device capabilities

Kernel selection is driven by declared device capabilities, not by backend name tests
scattered through the layer builders.

| Capability | Meaning |
| --- | --- |
| `PACKED_HALF2_MATH` | packed FP16 pair arithmetic holds CPU parity here |
| `WARP_SHUFFLE` | warp-level shuffle reductions |
| `SUBGROUP_SHUFFLE_32` | 32-wide subgroup shuffle (Metal's SIMD32 reductions) |
| `TENSOR_CORE_MMA` | tensor-core MMA kernels, which exist only on CUDA |
| `SPLIT_KV_ATTENTION` | split-KV flash attention |
| `SINGLE_PASS_RMS` | single-pass RMS normalization |

A capability that is withheld is withheld deliberately and is a divergence, not a bug:
`TENSOR_CORE_MMA` is CUDA-only because the MMA kernels do not exist elsewhere — the non-MMA
sibling is what the other backends run.

`SPLIT_KV_ATTENTION` is granted everywhere. The FP32 split-KV kernel gives each thread a
128-float accumulator row in local memory, 34052 bytes for its 64-thread workgroups, and Apple
GPUs allow 32 KB per threadgroup, so Metal used to refuse its pipeline and the capability was
withheld. `SplitKvAttentionPolicy` now reads the device's local-memory size and selects
`processHeadsFlashAttentionSplitKVPaged32`, the same arithmetic in 32-thread workgroups
(17284 bytes), wherever the 64-thread arrays do not fit.

On Metal, every family below prefills in batches of 512 by default
(`BatchPrefillSupport.defaultFor`), when the caller chose no execution policy and neither
`--with-prefill-decode` nor `-Djitllm.prefillBatchSize` was given: Qwen3, Llama (including
Q4_0), Qwen2 / DeepSeek-R1-Distill-Qwen, Mistral, Phi-3, Granite 3.2 / 4.0, Qwen3.5 (Q4_0)
and Gemma 4 (Q8_0). With MLX installed, the families MLX covers run that prefill's attention
and projections on MLX by default (see native libraries below). Pass `--no-batch-prefill`
(`-Djitllm.withPrefillDecode=false`) to keep single-token prefill.

The batched projections run on Apple's SIMD-group matrices (`simdgroup_matrix`, through
TornadoVM's `KernelContext` matrix API): `TransformerBatchPrefillSimdgroupKernels` stages
64-token by 64-row tiles of activations and weights in threadgroup memory, dequantizing Q8_0
and Q4_0 weights to `float` as they are staged, and multiplies them as 8x8 fragments. Fused
variants cover Phi-3's packed QKV and gate/up weights and Granite's scaled residual; strided
variants with FP16 activations write Gemma 4's packed QKV and gate/up rows. Attention over the
FP32 cache runs on SIMD-group matrices too: `TransformerPagedKvBatchPrefillSimdgroupKernels`
for 64-, 96- and 128-wide heads, and `Gemma4BatchPrefillSimdgroupAttentionKernels` for Gemma
4's 256- and 512-wide heads; over the FP16 cache (the default) their twins
`TransformerPagedKvBatchPrefillSimdgroupFP16Kernels` and
`Gemma4BatchPrefillSimdgroupAttentionFP16Kernels` load key and value fragments as
`simdgroup_half8x8`. The FP16 key/value cache is supported on Metal for every family above
except Gemma 4 Q4_0, in single-token and batched prefill (and sequential prefill/decode for
Llama, Qwen3 and Qwen3.5); rows, parity and FP16/FP32 speed are in
[kv-cache-support.md](kv-cache-support.md#metal). The SIMD-group kernels need TornadoVM's half-precision fragments,
transposed loads and vector-width array loads (`getFloat4` / `getHalf4`).
`-Djitllm.metal.simdgroupGemm=false` and `-Djitllm.metal.simdgroupAttention=false` select the
scalar kernels instead, where a family has them.

Prompt processing on an Apple M4 Pro (pp512, chunk 256, tok/s), single-token prefill against
batched; greedy text is identical in every row:

| Model | Single-token | Batched |
| --- | --- | --- |
| Qwen3-0.6B F16 / Q8_0 | 151 / — | 4171 / 3802 |
| Llama-3.2-1B F16 / Q8_0 / Q4_0 | 40 / 45 / 46 | 2372 / 2075 / 2124 |
| Qwen2.5-1.5B F16 / Q8_0 | — | 1708 / 1532 |
| DeepSeek-R1-Distill-Qwen-1.5B Q8_0 | — | 1498 |
| Mistral-7B Q8_0 | — | 304 |
| Phi-3-mini F16 / Q8_0 | 19 / 23 | 638 / 569 |
| Granite-3.2-2B F16 / Q8_0 | 33.5 / 43.6 | 950 / 854 |
| Granite-4.0-1B Q8_0 | 25.7 | 1378 |
| Qwen3.5-0.8B Q4_0 | 60 | 2057 |
| Gemma-4-E2B Q8_0 | 69 | 1067 |

On CUDA every family has a batched prefill on the tensor cores: Llama (F16, Q8_0, Q4_0),
Mistral, Qwen2 / DeepSeek-R1-Distill-Qwen, Qwen3, Phi-3, Granite, Devstral (F16, Q8_0, Q4_K),
Qwen2-MoE, Gemma 4, Qwen3.5 / 3.6 / 3.8 and deepseek2. It is the default,
in chunks of 512, wherever the family's batched prefill builds on the device; elsewhere the
single-token plan runs. `--with-prefill-decode --batch-prefill-size N` chooses the chunk, and an
explicit request the device cannot build is refused with `GPUL-CFG-002`, naming the flag to drop. The Llama-shaped families share
`LlamaFP16LayersBatchPrefillMMA` and `LlamaQ8_0LayersBatchPrefillMMA`, which a family extends
through hooks (its weights, its QKV and gate/up projections, what runs after the QKV
projection, its RoPE) rather than copying the layer graph.

- **Attention** (`TensorCoreBatchAttention`) gathers the layer's paged FP16 cache per key/value
  head, then runs Q·Kᵀ and P·V as FP16 GEMMs on the tensor cores with a causal softmax between
  them, a few query heads per pass so the scores fit in 64 MB. A head narrower than the GEMM's
  128-column tile (64, 96) is computed 128 wide and narrowed on the way to FP16. It needs the
  FP16 cache; with `--fp32-kv-cache` the scalar flash-attention kernel runs instead, about 4×
  slower.
- **Quantized projections** (`FP16Projections`) decode Q8_0 weights to FP16 and multiply them
  with the FP16 GEMM. Projections wide enough to fill the device are decoded in the tile order
  `gemmMMATiledB` stages and copied with `cp.async`. When the model's decoded weights fit in
  6 GB, each projection keeps its own decoded copy, decoded on the first chunk only — unless
  the load's budget has no room for one copy per session, when the preflight settles the
  policy's `decodedProjections` on a per-chunk scratch instead (pinned with
  `-Djitllm.prefill.decodedProjections=resident|per-chunk`).
  `DequantizingFP16LayersBatchPrefillMMA` does the same for Q4_0 / Q4_1 (packed tiles
  included) and Q4_K / Q6_K, decoding into one shared scratch per chunk.
- **Narrow projections** (`SplitKGemm`) whose output fills fewer than 128 tiles of 128×128 split
  their depth into two to four slices and sum the partial products afterwards. The summation
  order differs from the unsplit GEMM; `-Djitllm.prefill.splitK=false` keeps every GEMM whole.
- **Native libraries.** On by default where implemented and installed
  (`NativeLibrarySupport.defaultFor`): every one of these GEMMs is then
  `cublasGemmExFP16FP32`; Q, K and V (and gate and up) are stacked into one buffer so one call
  covers them. The flag is accepted for Llama, Mistral, Qwen2 / DeepSeek, Qwen3, Phi-3 and
  Granite (Q8_0, F16), Devstral (Q8_0, Q4_K, F16), Gemma 4 (Q8_0, Q4_0) and Qwen3.5 (Q8_0); the
  JIT kernels run elsewhere. Where the path exists but cuBLAS does not load, the run notes once
  (`For maximum performance please install cuBLAS 12 and cuDNN 9; falling back to the JIT
  kernels.`) and runs the JIT kernels. `--with-native-libraries` requires the path and refuses
  any configuration without one; `--no-native-libraries` keeps the JIT kernels. The TornadoVM binding loads `libcublasLt.so.12`. For Qwen3 F16 the first
  chunk's attention runs as a cuDNN fused SDPA when the TornadoVM cuDNN shim is built against
  cuDNN 9; against cuDNN 8.9 the plan cannot be built and the tensor-core GEMM attention runs.
- **Native libraries on Metal (Apple MLX).** With `--with-native-libraries` on a Metal device
  the batched prefill runs every chunk's attention as MLX's fused causal attention over the
  chunk's context, gathered from the paged cache into contiguous FP16 buffers, and its four
  projections as MLX GEMMs, launched by TornadoVM's `tornado-mlx` library tasks on TornadoVM's
  own queue and buffers (`MlxPrefill`). Decode stays on the JIT kernels. Each provider declares
  the weights it implements per backend (`TornadoPlanProvider.nativeLibraryWeights(BackendId)`):
  on Metal Llama (F16, Q8_0, Q4_0), and Mistral, Qwen2 / DeepSeek, Qwen3, Phi-3 and Granite
  (F16, Q8_0); everything else is refused. F16 weights are multiplied as they are, split Q/K/V
  and gate/up stacked into one copy each. Q8_0 and Q4_0 weights are either decoded once into
  resident FP16 copies or repacked exactly into MLX's affine format — the same
  `decodedProjections` residency as above, settled against the budget at load or pinned with
  `-Djitllm.prefill.decodedProjections=resident|per-chunk`. Packed projections are refused by
  name. Needs `brew install mlx` (MLX 0.32.3) and an M4-class GPU; validated with
  `-Dtornado.mlx.async=true`, without which every MLX call waits for the GPU and prefill is
  slower than the JIT kernels.

Prompt processing on an NVIDIA A100 80GB (pp512, chunk 512, `--cuda-graphs`, tok/s); batched
greedy text matches single-token text except where noted:

| Model | Single-token | Batched | Batched, native libraries | llama.cpp b11382 |
| --- | --- | --- | --- | --- |
| Llama-3.2-1B Q8_0 | 447 | 16,774 | 28,929 | 23,550 |
| Llama-3.2-1B Q4_0 | 382 | 15,299 | 25,745 | 22,626 |
| Llama-3.2-1B F16 | 266 | 16,757 | 31,852 | 41,331 |
| Qwen3-0.6B F16 (native: cuBLAS + cuDNN 9 attention) | — | 23,804 | 54,355 | 40,522 |
| Qwen2.5-0.5B Q8_0 | — | 23,046 | 33,503 | 31,630 |
| DeepSeek-R1-Distill-Qwen-1.5B Q8_0 | — | 13,797 | 18,560 | 16,207 |
| Granite-3.2-2B Q8_0 | — | 7,490 | 13,639 | 9,814 |
| Phi-3-mini Q8_0 (split-K reorders sums; text drifts after ~20 tokens) | — | 5,384 | 8,870 | 8,234 |
| Mistral-7B Q8_0 | — | 2,947 | 5,522 | 4,600 |
| Devstral-24B Q4_K_M | 4.7 | 836 | 1,333 | 1,451 |
| Llama-3.3-70B Q4_0 | 20 | 324 | 664 | 556 |

`PACKED_HALF2_MATH` is withheld on OpenCL. Packed FP16 arithmetic rounds each product to
FP16 before the FP32 accumulator sees it, once per term; over a 2048-term projection row
that loss is systematic rather than cancelling, and on OpenCL it was large enough to fail
CPU parity for the Llama-shaped FP16 QKV projection while the identical kernel held on CUDA.
Where the capability is withheld, the projection widens each pair before multiplying.

The capability is narrower than its name suggests, and the distinction matters. Other
kernels use packed pairs and are unaffected on OpenCL — `fusedRmsNormFFNGateUp`, shared with
Qwen3, holds parity there. What is specific to the QKV projection is that **both** operands
are FP16: the weights, and an activation that `mapContextWithQuantize` has already rounded
to FP16. That is a double rounding before an FP16 multiply, and it is the combination, not
packed arithmetic on its own, that loses the accuracy. Metal keeps the packed path and has
not been evaluated against this; a Metal FP16 investigation should measure it before
assuming the CUDA result carries over.

## What each backend supports

| Capability | CPU | CUDA | OpenCL | Metal |
| --- | --- | --- | --- | --- |
| Model loading, device selection | yes | yes | yes | yes |
| `STANDARD` single-token decode | yes | yes | yes | yes |
| Sequential prefill/decode | yes | yes | yes | yes |
| Batched prefill/decode | yes | yes (tensor cores, every family; cuBLAS where installed) | yes | yes (SIMD-group kernels, FP16 or FP32 cache; MLX where installed) |
| FP16 key/value cache (the default) | yes | yes | NVIDIA-class devices | yes, except Gemma 4 Q4_0 ([matrix](kv-cache-support.md#metal)) |
| F16 and Q8_0 weights | yes | yes | yes | yes |
| Q4_K / Q6_K device residency | n/a | yes | yes | yes |
| Q4_0 device residency (Llama) | n/a | yes | yes | yes |
| CPU-resident sampling | yes | yes | yes | yes |
| Device-resident sampling | n/a | yes | yes | yes |
| Shared KV pool and leases | yes | yes | yes | yes |
| Prefix caching | yes | yes | yes | yes |
| Compiled-program caching | n/a | yes | yes | yes |
| Lowered execution path under `auto` | n/a | Llama/F16/`STANDARD` | selects legacy | selects legacy |
| `qwen35` (Qwen3.5 / 3.8) | yes | all three modes, Q4_0 | untested | single-token and batched prefill, Q4_0 |
| Conversations, tools, thinking control, streaming | yes | yes | yes | yes |
| Memory preflight confidence | n/a | `EXACT` | `EXACT` | capped at `CONSERVATIVE` |
| Reset / close / multi-session | yes | yes | yes | yes |

Recorded external limitations, each with its named cause:

- **Gemma 4 Q4_0 batched prefill on Metal** — refused by `BatchPrefillSupport`: the Metal
  SIMD-group projections decode Q8_0 blocks for this family, and the Q4_0 file also carries
  Q4_1 tensors. The tensor-core kernels (`gemmMMAQKV` and siblings) remain CUDA-only:
  `TornadoInternalError: unimplemented: MMA instructions only supported for the PTX backend`
  (TornadoVM's message still names PTX, the assembly the CUDA backend emits).
- **Qwen3 on Apple's OpenCL** — `clCreateKernel(processHeadsFlashAttentionSplitKVPaged)
  failed: CL error -48`, in every configuration and both representations. The same model is
  correct on Metal. Apple's OpenCL-over-Metal shim; Linux OpenCL is unaffected.
- **Kernel capture on Metal** — `withPrintKernel()` produces no kernel source, so
  `CompiledProgramIdentityAccelTest` cannot observe there. A capture-path gap, not a
  numerical one.
- **`qwen35` runs all three modes on CUDA; single-token and batched prefill on Metal.** Sequential prefill reuses the
  single-token layer graphs with the logits graph skipped; batched prefill has its own layer
  graphs, in which the convolution and the delta rule scan the chunk in token order inside the
  kernel rather than treating its rows as independent. The MTP draft head is not built into any
  of them. On Metal the batched Q4_0 projections run on the SIMD-group GEMMs and decode
  attention takes 8-position key/value tiles (`processHeadsFlashAttentionPagedTile8`): with
  16-position tiles a 256-wide head needs 33.8 KB of threadgroup memory, over Metal's 32 KB.
  OpenCL has not been run.
- **`qwen35` attention does not use the split-KV kernel.** Its head is 256 wide and
  `processHeadsFlashAttentionSplitKVPaged` fixes its query staging and per-thread accumulator
  at 128 floats per head, so a 256-wide head reads and writes past them — on CUDA an illegal
  address, which surfaces as a poisoned context and an allocation failure in an unrelated
  call. This family uses the single-workgroup online-softmax kernel, which sizes its shared
  memory from the head width it is given, at the cost of the splits' parallelism.
- **Memory preflight on Metal** is capped at `CONSERVATIVE`. The multiplicity/header model
  `EXACT` depends on was bisected against measurement on CUDA only, and admission acts on
  the confidence level.

Batched prefill is **the default on every accelerator backend** where the family builds it: in
chunks of 512 on CUDA, OpenCL and, for the families listed above, Metal. A model
split across devices keeps single-token prefill unless batches are asked for.

## Metal specifics

Metal is a first-class backend, verified on Apple silicon by the same gates as the others
where the fixtures exist there. Two things about the toolchain are worth knowing because
they cost real time otherwise:

- **`make metal` in TornadoVM is not Metal-only** — it expands to `--backend metal,opencl`.
  Use `make BACKEND=metal`. On a multi-backend SDK, TornadoVM's backend tie-break can send
  a run to Apple's OpenCL, where a Qwen3 kernel will not compile and healthy numbers look
  like regressions.
- Metal's single-token kernel selection differs from CUDA/OpenCL through
  `SUBGROUP_SHUFFLE_32` and the device's 32 KB of threadgroup memory, which selects the
  32-thread split-KV kernel and Qwen3.5's 8-position attention tiles. Batched prefill
  selects the SIMD-group kernels on Metal (`BatchPrefillGemmPolicy`).
- **Metal compiles kernels with fast math**, and its fast `tanh` returns NaN for large
  arguments. Gemma 4's GELU once turned a single large feed-forward activation into NaN that
  reached every later layer and token, so the model printed nothing. `geluActivation` clamps
  the `tanh` argument to ±10, beyond which `tanh` is already exactly ±1 in `float`, so other
  backends compute the same values.
- Granite's single-token attention on Metal takes the flash kernel:
  `processHeadsParallelGranitePaged` gives wrong output there.

Qwen2 and Qwen3 in F16 once produced fluent-looking token salad on Metal, in every
execution mode, while exiting 0 and reporting normal throughput; the cause was
capability-gated kernel selection, and the fix routes them to the verified SIMD32 kernels.

**Qwen2.5 Q8_0 on Metal is still wrong** in the same way — correct backend, real execution
path, exit code 0, and backticks instead of an answer. It is an open defect, not a recorded
limitation. Both are in [`verification.md`](verification.md), because the lesson is about
how they were measured rather than about Metal.
