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
`SPLIT_KV_ATTENTION` is not granted to Metal, whose driver refuses to JIT
`processHeadsFlashAttentionSplitKV`, and `TENSOR_CORE_MMA` is CUDA-only because the MMA
kernels do not exist elsewhere — the non-MMA sibling is what the other backends run.

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
| Batched prefill/decode | yes | yes (F16; Q8_0 blocked, see below) | yes | **blocked** |
| F16 and Q8_0 weights | yes | yes | yes | yes |
| Q4_K / Q6_K device residency | n/a | yes | yes | yes |
| Q4_0 device residency (Llama) | n/a | yes | yes | yes |
| CPU-resident sampling | yes | yes | yes | yes |
| Device-resident sampling | n/a | yes | yes | yes |
| Shared KV pool and leases | yes | yes | yes | yes |
| Prefix caching | yes | yes | yes | yes |
| Compiled-program caching | n/a | yes | yes | yes |
| Lowered execution path under `auto` | n/a | Llama/F16/`STANDARD` | selects legacy | selects legacy |
| `qwen35` (Qwen3.5 / 3.8) | yes | all three modes, Q4_0 | untested | untested |
| Conversations, tools, thinking control, streaming | yes | yes | yes | yes |
| Memory preflight confidence | n/a | `EXACT` | `EXACT` | capped at `CONSERVATIVE` |
| Reset / close / multi-session | yes | yes | yes | yes |

Recorded external limitations, each with its named cause:

- **Batched prefill on Metal** — TornadoVM lowers the MMA batch-prefill kernels
  (`gemmMMAQKV` and siblings) only on CUDA:
  `TornadoInternalError: unimplemented: MMA instructions only supported for the PTX backend`
  (TornadoVM's message still names PTX, the assembly the CUDA backend emits).
- **Q8_0 batched prefill on CUDA** — a TornadoVM CUDA address-lowering gap on the Q8_0
  tensor-core kernel: `TornadoInternalError: unimplemented: address origin unimplemented`
  in `CUDAAddressLowering.lower`. Reported upstream as beehive-lab/TornadoVM#1057.
- **Qwen3 on Apple's OpenCL** — `clCreateKernel(processHeadsFlashAttentionSplitKVPaged)
  failed: CL error -48`, in every configuration and both representations. The same model is
  correct on Metal. Apple's OpenCL-over-Metal shim; Linux OpenCL is unaffected.
- **Kernel capture on Metal** — `withPrintKernel()` produces no kernel source, so
  `CompiledProgramIdentityAccelTest` cannot observe there. A capture-path gap, not a
  numerical one.
- **`qwen35` runs all three modes on CUDA, and only on CUDA.** Sequential prefill reuses the
  single-token layer graphs with the logits graph skipped; batched prefill has its own layer
  graphs, in which the convolution and the delta rule scan the chunk in token order inside the
  kernel rather than treating its rows as independent. The MTP draft head is not built into any
  of them. OpenCL and Metal have not been run.
- **`qwen35` attention does not use the split-KV kernel.** Its head is 256 wide and
  `processHeadsFlashAttentionSplitKVPaged` fixes its query staging and per-thread accumulator
  at 128 floats per head, so a 256-wide head reads and writes past them — on CUDA an illegal
  address, which surfaces as a poisoned context and an allocation failure in an unrelated
  call. This family uses the single-workgroup online-softmax kernel, which sizes its shared
  memory from the head width it is given, at the cost of the splits' parallelism.
- **Memory preflight on Metal** is capped at `CONSERVATIVE`. The multiplicity/header model
  `EXACT` depends on was bisected against measurement on CUDA only, and admission acts on
  the confidence level.

Batched prefill stays **default-off on every backend** regardless of availability.

## Metal specifics

Metal is a first-class backend, verified on Apple silicon by the same gates as the others
where the fixtures exist there. Two things about the toolchain are worth knowing because
they cost real time otherwise:

- **`make metal` in TornadoVM is not Metal-only** — it expands to `--backend metal,opencl`.
  Use `make BACKEND=metal`. On a multi-backend SDK, TornadoVM's backend tie-break can send
  a run to Apple's OpenCL, where a Qwen3 kernel will not compile and healthy numbers look
  like regressions.
- Metal's kernel selection differs from CUDA/OpenCL only through `SUBGROUP_SHUFFLE_32` and
  the withheld `SPLIT_KV_ATTENTION`. Nothing else branches on the backend.

Qwen2 and Qwen3 in F16 once produced fluent-looking token salad on Metal, in every
execution mode, while exiting 0 and reporting normal throughput; the cause was
capability-gated kernel selection, and the fix routes them to the verified SIMD32 kernels.

**Qwen2.5 Q8_0 on Metal is still wrong** in the same way — correct backend, real execution
path, exit code 0, and backticks instead of an answer. It is an open defect, not a recorded
limitation. Both are in [`verification.md`](verification.md), because the lesson is about
how they were measured rather than about Metal.
