---
name: port-model-to-jitllm
description: Port a new model architecture to jitllm. Use when a GGUF file is not recognized, when a family needs its own program description or lowering, or when deciding whether an existing family's implementation can be reused for a new one.
license: MIT
---

# Port a model architecture to jitllm

A port is: recognize the file, describe the computation once in the neutral operation
vocabulary, implement it on the CPU as the reference, let each backend lower it, and prove
the numbers against that reference on real hardware.

Read [`docs/architecture/architecture.md`](../../../docs/architecture/architecture.md) and
[`docs/architecture/models-and-backends.md`](../../../docs/architecture/models-and-backends.md)
first. This skill is the workflow; those are the contracts.

Qwen 3.8 is the motivating example throughout. Nothing here implements it.

## 1. Inventory before editing anything

Write the inventory down before touching code. Half the traps below are questions this
step answers.

**From the GGUF metadata:**

- `general.architecture` and `general.name`. These decide recognition, and they are not
  always what the model is called: Mistral and older Devstral builds declare `llama`, a
  DeepSeek distill declares `qwen2`, and newer Devstral declares `mistral3`.
- the `<arch>.*` configuration block: layer count, embedding length, head counts, head
  dimension, feed-forward length, context length, norm epsilon, RoPE base and scaling.
- tokenizer model, merges, special and terminal token ids, and the chat template.

**From the tensor table:**

- every tensor name, shape, dtype and quantization;
- which tensors are per-layer and which are global;
- whether output is tied to the token embedding, and whether the two are separate entries
  over the same storage;
- biases: which projections have them, which do not;
- fused layouts — a single QKV tensor, a single gate/up tensor — and the exact interleaving;
- MoE: router, per-expert and shared-expert tensors, and whether experts are stacked into
  one tensor or stored separately.

**From the architecture itself:**

- attention geometry: query heads, key/value heads, and the head dimension. **Do not assume
  `dim / heads`** — several families state it independently.
- RoPE variant and layout (interleaved vs. neox halves), base frequency, and any scaling;
- normalization: which points, RMS or layer norm, and any per-head query/key norm;
- FFN activation: SwiGLU, GeGLU, or something else;
- any logit soft-cap, attention scaling or scale factor the family applies;
- sliding-window attention, or any KV reuse across layers — these change the program's
  topology, not a parameter of attention.

**And the target:**

- which execution modes you intend to support: `STANDARD`, `PREFILL_DECODE`,
  `BATCH_PREFILL_DECODE`;
- which dtypes: F16, Q8_0, and which K-quants;
- which backends: CPU, CUDA, OpenCL, Metal, and any capability a kernel choice will need.

A useful first command, on a real file:

```bash
JAVA_TOOL_OPTIONS="-Djitllm.metrics.format=json -Djitllm.metrics.output=stdout" \
  ./jitllm --model <model.gguf> --prompt hi -n 1 --verbose-init
```

An unrecognized file fails with `[GPUL-MOD-002]` and prints the declared architecture and
name, which is the first two inventory lines for free.

## 2. Decide whether the existing abstractions already express it

Walk these in order and answer each with yes, or with what is missing.

| Extension point | Question |
| --- | --- |
| `ModelProvider` | can `GgufRecognition` map this file to an `ArchitectureId`, and can a loader read its metadata block? |
| `ModelArchitecture` | can `describe(...)` build the forward pass from existing `OperationKind` values over existing `TensorRole` values? |
| `CpuForwardProvider` | can the CPU path be composed from `CpuOperations`, or does it need new arithmetic? |
| `TornadoPlanProvider` | which `DataType`s and `ExecutionMode`s will it declare, and do layer graphs exist for them? |
| `TornadoLoweringProvider` | is a lowering needed at all, or does the family run legacy only? |
| `KvStorageFactory` | is the KV layout one an existing factory produces? |
| `TensorRole` | is every weight expressible as an existing role? |
| `DataType` | are the file's quantizations already in the vocabulary, and does the target backend accept them? |
| state and workspace | are the activation, attention and staging buffers the same shapes an existing family allocates? |

If every answer is yes, this is a composition exercise and section 4 is the whole job.

## 3. Stop before writing code when the model needs a new concept

**Stop and write a short design proposal** when the port requires any of:

- a new public API concept, or any change to a frozen facade type;
- a new `OperationKind` or `TensorRole`;
- a tensor layout or view the descriptor vocabulary cannot express;
- new KV ownership, a new layer topology, or KV shared across layers;
- a kernel rewritten rather than reused;
- a backend-specific fact appearing in a neutral package.

The proposal is short and says five things: **the smallest new concept** that covers the
need, the **alternatives** considered and why they are worse, **compatibility** with the
families that already exist, the **backends affected** and what each must implement, and the
**verification plan** that will show it correct.

**Do not invent a family-specific composite to avoid extending the vocabulary.** An
operation named after one model, doing what two existing operations do, is how the vocabulary
stops being shared — and the next family gets a second one. If two families need the same
arithmetic, that is one operation with two callers.

Equally: do not extend the vocabulary for something already expressible. A fused QKV
projection is `SPLIT_FUSED_QKV` over an `ATTENTION_QKV` weight, not a new operation.

## 4. Implement in dependency order

Each step is testable before the next one exists. Do them in this order, because a defect
found later is a defect in something already believed correct.

1. **Recognition and metadata validation.** Add the case to `GgufRecognition`, returning
   this model's own `ArchitectureId`. Add a `ModelProvider` whose `supports` and
   `architecture` agree with it, and a `validateConfiguration` on the `ModelArchitecture`
   that rejects a metadata block that cannot be run — before any weight is read.
2. **Configuration, tokenizer, chat format.** The chat template, special tokens and stop
   tokens are the family's, and they stay internal: nothing about them reaches the facade.
3. **Tensor descriptors and loading.** Map each tensor to a `TensorRole`. Materialize into
   the representation the chosen backend executes. Decoding a quantized tensor is a loading
   concern; it is not a kernel's job and not a model's.
4. **CPU reference path.** `CpuForwardProvider` plus a `ForwardPass` composed from
   `CpuOperations`. This is the reference every later number is judged against, so it comes
   before any accelerator work. Add a `*CpuOperationEquivalenceTest`-style unit test over a
   tiny synthetic model: it needs no GGUF, no device, and it pins the decomposition.
5. **Logical program description.** `ModelArchitecture.describe(...)` returning an
   `InferenceProgram`: ordered components over named weights, declared phases, and a
   signature. No device handles, no task graphs.
6. **Backend support and lowering.** `TornadoPlanProvider` declaring the dtypes and modes
   that actually run, then a `TornadoLoweringProvider` if the family is to lower. Declare
   only what the selection layer can execute.
7. **State, workspace and KV.** A `KvStorageFactory` if the layout differs, and whatever
   the state must allocate. Anything sized at allocation must be known at allocation.
8. **CLI and facade reachability.** The model loads through `LocalModels.load`, opens a
   session, and generates — with no new public type and no family name in any signature.

Register each provider in `src/main/resources/META-INF/services/`. There is no central
switch to edit, and adding one fails the architecture rules.

## 5. Backend neutrality is checked, not trusted

- No TornadoVM type outside `backend/tornado` (rule 1), and no `TaskGraph`,
  `GridScheduler`, `ImmutableTaskGraph` or `TornadoExecutionPlan` outside it (rule 11).
- No GGUF or file-format type in a backend contract or the runtime tensor vocabulary
  (rule 4).
- No model architecture package importing TornadoVM (rule 2).
- Fusion and kernel choice belong to the backend. The program says what is computed.
- An architecture, dtype or mode combination that is not supported **fails by name**:
  `UnsupportedLoweringException` for an unimplemented lowering under `jitllm.lowering=on`,
  `UnsupportedOperationException` for a device selector this build cannot honour. Never
  substitute silently, and never convert a representation a backend did not accept.

`./mvnw test` runs the rules. A violation is a build failure, not a review comment.

## 6. Verify against a real fixture

A synthetic fixture proves the decomposition. Only a real one proves the port.

- **Fixture identity.** Record the file's SHA-256 and where it came from. Add it to
  `GoldenFixture` so the suite can find it under `$JITLLM_TEST_MODELS` or
  `~/.jitllm/test-models/`, and so an absent fixture skips with a named reason instead
  of passing.
- **Deterministic CPU reference.** Generate on the CPU path, twice, and confirm the two
  agree before comparing anything to them.
- **CPU/accelerator parity.** Add a `<Family>CpuGpuParityAccelTest` extending `CpuGpuParity`,
  with a case per representation. One class per family, because surefire forks per class and
  a class that loads every fixture exhausts the device partway through. This is the gate that
  matters: the CPU is the reference, and it is the only comparison that can see a defect which
  moves the whole GPU.
- **Legacy versus lowered**, where both exist. Bit identity where the paths are meant to be
  identical; the parity bounds where they are not. This is a *supplementary* check, never
  the primary one.
- **Golden logits** only once the tuple is pinned and reproducibility has been measured
  rather than assumed. Never regenerate a golden to make a run pass.
- **Token-level smoke.** A prompt whose correct answer is a substring, asserted on the
  generated text.
- **Lifecycle.** Reset, multi-turn recall, close, and use-after-close.
- **Every mode declared.** If the provider says `PREFILL_DECODE`, run it and show it is
  token-identical to `STANDARD`. If it says `BATCH_PREFILL_DECODE`, run that too.
- **Every dtype declared.** F16 and each quantization the provider accepts.
- **The standing gates:** `DecodeAllocationAccelTest` (no per-token allocation),
  `CompiledProgramIdentityAccelTest` (compiled once, unchanged after warm-up),
  `MemoryPlanAccuracyAccelTest` and `GraphTopologyConsistencyTest` (the memory plan matches
  the topology), and the benchmark gate for a performance claim.
- **Each backend separately** — CUDA, OpenCL, Metal — with anything unsupported recorded by
  name and cause, not omitted.

Every accelerator run must prove three things before its result counts: the resolved
backend, a real `execution_path` in the metrics JSON, and correct output. See
`build-n-run-engine`.

## 7. Aliases and shared implementations

A family may reuse another's computation. It may not borrow its identity.

- **Keep the model's own `ArchitectureId`.** It is what the provider declares, what the
  program signature carries, and what the compiled-program cache keys on. Two models sharing
  an id share cache entries.
- **Reuse is a verified claim.** Running Qwen 3.8 on Qwen3's program description is fine
  *after* CPU/accelerator parity passes for Qwen 3.8's own fixture — not because the
  configuration classes look alike.
- **Never infer identity from a shared configuration class.** A shared `Configuration`
  implementation says two families have the same fields, nothing more.
- **No central switch.** Discovery is `ServiceLoader`; a `switch` on model type fails
  rule 15.

## 8. Finish the port

- Update the family list and the per-backend table in
  [`docs/architecture/models-and-backends.md`](../../../docs/architecture/models-and-backends.md),
  and the known limitations in
  [`docs/architecture/verification.md`](../../../docs/architecture/verification.md).
- Register every provider in `META-INF/services/`, and check the built jar carries them:
  `jar tf target/jitllm-*.jar | grep META-INF/services/org.beehive.` — CI asserts the
  count, because a missing entry loses the family silently at runtime.
- Run the architecture rules and the addition-workflow tests: `./mvnw test`.
- Add the model's rows to `.github/workflows/standalone-inference.yml`, and add an entry to
  `.github/standalone-expectations.tsv` for any combination that is genuinely blocked —
  with its *observed* cause. A row producing wrong output is a defect, and the assertion
  step refuses to let the table silence it.
- Keep GGUF files, profiler output and raw performance data untracked.
- Report exactly what ran on real hardware, on which device and SDK, and what did not.

## Checklist

```
[ ] Inventory written down: metadata, tensors, geometry, tokenizer, targets
[ ] Every extension point answered yes, or a design proposal written
[ ] Recognition returns this model's own ArchitectureId
[ ] validateConfiguration rejects an unrunnable metadata block before loading weights
[ ] Tensor roles assigned; fused and tied layouts stated explicitly
[ ] CPU reference implemented and pinned by a synthetic-model unit test
[ ] Program description composed from existing operations
[ ] Provider declares only the dtypes and modes that execute
[ ] KV storage, state and workspace requirements met at allocation time
[ ] Services registered and present in the built jar
[ ] Architecture rules pass
[ ] Real fixture registered with its hash
[ ] CPU/accelerator parity passes for every declared dtype
[ ] Every declared execution mode run
[ ] Lifecycle: reset, multi-turn, close, use-after-close
[ ] Allocation, program-identity and memory-plan gates pass
[ ] Each backend run separately; unsupported combinations named with causes
[ ] Docs and CI matrices updated
[ ] Nothing large or generated is tracked
```

## Common traps

**A GPU-versus-GPU comparison cannot see a defect that moves the whole GPU.** Lowered and
legacy paths that are equally wrong agree with each other and both pass. Score against the
CPU.

**Exit code 0 and fluent text prove nothing.** An accelerator computing entirely wrong
numbers produces both, at a normal throughput. A matrix that scored rows on `achieved tok/s`
once called three token-salad rows passing.

**An alias can inherit the wrong identity.** Recognizing a new model by mapping it onto an
existing `ArchitectureId` makes it share that family's cache entries and providers. Give it
its own id and let it reuse the implementation explicitly.

**A provider can declare a mode the selection layer cannot run.** `supportedModes()` is a
claim. If nothing builds layer graphs for that mode, the failure arrives from inside
TornadoVM. Declare what executes, and let `TornadoBackendSupportTest` hold the set.

**An invalid synthetic fixture fails both paths identically**, which looks like agreement.
If a reference and a candidate both throw, the test has proved nothing — check the fixture
before the code.

**Tied weights may be two wrappers over one storage.** The output projection and the token
embedding can be separate entries backed by the same bytes. Treat them as one allocation and
two roles, not two allocations.

**Head dimension is not always `dim / heads`.** Several families state it independently, and
using the quotient silently produces the wrong stride.

**Fused QKV and fused gate/up need their interleaving stated.** Splitting on the wrong
boundary produces plausible text and wrong logits — the failure mode the CPU parity gate
exists to catch.

**KV reuse across layers changes the program's topology**, not an attention parameter. It
changes what the layer graph binds and how many buffers the memory plan predicts.

**Do not infer a backend capability from a vendor name when a capability exists.** Kernel
selection branches on `DeviceCapability`. A test on the backend's name is how a kernel that
is correct on one device gets selected on another where it is not.

**A batched kernel that re-reads the weights per row is not faster than running the rows one at a
time.** Batched prefill's first measurement on a 27B was *exactly* single-token throughput: one
workgroup per (row, output row) reads the weight matrix once per row, which is what the rows
separately read. Quantized projections are memory-bound, so a chunk only pays once it reuses what
it reads — decode a block once for a tile of rows. Tiling took prompt evaluation from 1.0x to
2.17x, and the per-row arithmetic is unchanged, so parity is unaffected.

**A worker grid keyed on a task name is wrong when one name maps to two representations.** {@code
ffn_down} is Q4_1 on a model's first blocks and Q4_0 on the rest, so the same task is tiled in one
layer and not in another. Key the grid on the layer's qualified task name, not the bare one.

**A prefill loop's token budget belongs to the caller.** Two families' decode loops disagreed about
what a prompt costs — the positions actually fed, or {@code promptTokens.size()} — and the prefill
loop guessed. The symptom is a *row count* one larger than the reference, not wrong numbers, so a
logits comparison reports it as a shape mismatch and every tolerance looks fine.

**A recurrence makes prompt tokens dependent inside a layer.** A chunk cannot be treated as
independent rows: the scan belongs inside the kernel, one lane per channel or per state column,
walking the chunk in order. Then the arithmetic per token is identical to the single-token kernel's
and the chunk width cannot change the answer — which is a property worth asserting directly, at
several widths and against each other.

**A shared kernel may have a fixed local-array size your model exceeds.** The split-KV attention
kernel stages a query and a per-thread accumulator in local arrays fixed at 128 floats per head.
A 256-wide head reads and writes past them; on CUDA that is an illegal address, and it surfaces as
a poisoned context and an allocation failure in an unrelated call several steps later, not as a
fault in the kernel that caused it. Read the constants in a kernel before reusing it at a new
geometry.

**A quantized row must be a whole number of blocks.** Every block-decoding kernel addresses a row
as `row * blocksPerRow` blocks. That holds for every projection a real quantizer produces, and
where it does not the kernel reads the next row's blocks and produces weights of plausible
magnitude. Check it at plan construction; the alternative is finding it as a numerical
disagreement.

**Declaring what a family reads per tensor is a different question from what a plan is selected
on.** A mixed model reports one representation and holds several. One set cannot answer both
without lying to the memory preflight, which then predicts every tensor at the model's dtype.

**A state that decides what to allocate from a system property will be wrong the moment a provider
exists.** Whether a session is on a device is known by the model, from its weights. A property is
set by one caller and not by the harness, and the failure is a null buffer inside the backend.

**Prove the path, every time.** An accelerator test that does not assert the resolved
backend and a real `execution_path` cannot tell a GPU run from a silent CPU fallback.
