# Architecture

How the engine is layered, which way dependencies point, and what the principal
abstractions are. [`execution.md`](execution.md) covers what happens when a model runs;
this document covers where the pieces live.

## Layers

```
                Integrations
      (CLI, OpenAI server, LangChain4j, Quarkus, user applications)
                     |
                     v
            Public API and generation
      (LocalModels, LocalModel, GenerationSession, stop conditions,
                  streaming, detokenization)
                     |
                     v
                   Engine
     (LLMEngine, Scheduler, admission, KvCacheManager, BlockPool,
                 PrefixCache, serving metrics)
                     |
                     v
              Models and sessions
     (loaded model: architecture + configuration + weights;
        session: sequence position + KV lease; provider SPI)
                     |
                     v
         Inference programs and operations
     (backend-neutral program description, program components,
              the shared operation vocabulary)
                     |
                     v
            Runtime, tensors and state
     (tensor descriptors, DataType, state layout, memory planning,
              execution policy, metrics sink)
                     |
                     v
                 Backend SPI
      (device selection, capabilities, compiled programs, buffer
                  lifetimes, capacity query)
                     |
                     v
        TornadoVM backend            CPU backend
   (task graphs, grid schedulers,   (plain Java, Vector API)
      execution plans, kernels)
                     |
                     v
       CUDA / OpenCL / Metal devices
```

Dependencies point **downward only**. A layer may depend on the layer below it and on
layers further below, never on a layer above. Sibling packages within a layer depend on
each other only through explicit interfaces.

Two edges deliberately run the other way:

- the **metrics sink** lives in the runtime layer, is written by backends, and is read by
  the engine and the API — the one designed upward-looking seam;
- **KV storage** is owned by a cache manager above the session and leased downward.

### Why the backend SPI sits below the runtime abstractions

Tensors, state and program descriptions have to be expressible without knowing which
backend will run them; that is what makes an inference program backend-neutral. The
backend SPI is the narrow interface through which those neutral descriptions become
executable. Backends implement the SPI, and nothing above the SPI knows they exist.

## Packages

| Package | Layer | Contents |
| --- | --- | --- |
| `api` | public API | the facade: `LocalModels`, `LocalModel`, `GenerationSession`, requests, results, events, options |
| `engine` | engine | `LLMEngine`, `Scheduler`, request handles and states |
| `model`, `model/<family>`, `model/architecture`, `model/provider`, `model/loader` | models | loaded models, family descriptions, the provider SPIs, GGUF loading |
| `inference`, `inference/state`, `inference/weights`, `inference/sampler`, `inference/op` | models/programs | generation loop, session state, weights, samplers, the CPU operation implementations |
| `program`, `program/op` | programs | the backend-neutral program model and its operation vocabulary |
| `runtime/*` | runtime | tensor descriptors and `DataType`, KV storage and leases, memory planning, execution policy, backend/device identity, metrics, diagnostics |
| `backend/cpu` | backend | the plain-Java forward passes |
| `backend/tornado` | backend | everything TornadoVM: task graphs, grid schedulers, kernels, lowering, device buffers |
| `format`, `tensor` | supporting | GGUF parsing and the host-side tensor representations |
| `tokenizer` | supporting | tokenizers and chat formats |
| `server`, `bench` | integrations | the OpenAI-compatible server and the benchmark harness |

## Enforced dependency rules

These are executable, not aspirational: `ArchRules` in `src/test/java/.../arch` states each
one as an ArchUnit rule and `DependencyRulesTest` runs them as an ordinary unit test. The
allowlists are shrink-only, and a stale allowlist entry fails the build.

| Rule | Statement |
| --- | --- |
| 1 | TornadoVM types appear only inside `backend/tornado`. The allowlist is **empty**. |
| 2 | Model architecture packages do not import TornadoVM. |
| 3 | The backend-neutral program layer imports neither TornadoVM nor any backend. |
| 4 | GGUF types stay in the format layer and the loading path; the runtime tensor vocabulary names no file-format type and no backend type. |
| 5 | Models own immutable configuration and weights, nothing mutable per sequence. |
| 7 | KV storage is never reachable from a model or a program. |
| 8a | Generation policy — the token loop, stop conditions, streaming — is separate from forward execution, and lower layers do not reach up into it. |
| 8b | Sampling is an operation and may execute on the device. It is not generation policy. |
| 11 | `TaskGraph`, `ImmutableTaskGraph`, `TornadoExecutionPlan` and `GridScheduler` stay in the backend. The allowlist is **empty**. |
| 14 | Core abstractions do not assume generation: nothing in the tensor, program or runtime layers requires a token loop to exist. |
| 15 | No central model-type switch. A new architecture is added by registering a provider, not by editing a `switch`. |
| 16 | No console I/O outside the CLI integration. |
| 17 | The metrics seam depends on nothing else in the project. |
| 18 | Nothing below the engine depends on the engine. |

Rule 14 and rule 18 together are what keeps non-generative use cases possible: an
embedding or classification pass needs a program, a backend and a device, and must not be
forced to acquire a sampler, a KV cache or a token loop.

## Provider discovery

Every extension point is a `ServiceLoader` SPI, declared in `META-INF/services`. Adding a
model family or a backend capability means adding a provider and its service entry; no
central registry or dispatch table is edited. The published jar carries seven service
files, and CI checks that count, because a missing one silently removes a model family.

| SPI | Answers |
| --- | --- |
| `ModelProvider` | does this GGUF belong to me, and how do I load it? |
| `ModelArchitecture` | what is this family's layer topology and program description? |
| `CpuForwardProvider` | how does this family run on the CPU? |
| `TornadoPlanProvider` | how does this family build TornadoVM task graphs? |
| `TornadoLoweringProvider` | how does this family lower to the shared workspace plan? |
| `KvStorageFactory` | how is this family's KV storage laid out? |
| `DeviceResolver` | which devices does this backend offer, and what can they do? |

Recognition is deliberately conservative. `GgufRecognition` maps a file's declared
`general.architecture` (plus, where the format leaves no other signal, its
`general.name`) to an `ArchitectureId`. An architecture nobody claims is reported as
unrecognized rather than guessed at — a near-miss family loads and then produces wrong
numbers, which is worse than not loading.

## Principal abstractions

**Loaded model** — configuration, weights, tokenizer, chat format and architecture
identity, immutable after load and shared by any number of sessions. It holds no sequence
position, no KV contents and no lease.

**Session** — one in-progress sequence. It owns its position, its KV lease, its sampler
and its conversation. It is not thread-safe; a loaded model is.

**Inference program** — a backend-neutral description of one forward pass: which
operations, over which weights, in what order, producing which outputs. Data, not
execution. It contains no device handles and no task graphs.

**Operation** — a reusable primitive with defined inputs and outputs: RMSNorm, RoPE,
matrix–vector multiply, attention, SwiGLU, softmax, residual add, sampling. An operation
says *what* is computed, never how it is scheduled.

**Compiled program** — a backend-specific executable form of an inference program together
with the device resources it needs. Built once, invoked many times, never rebuilt per
token.

**Backend** — an implementation that compiles inference programs, executes compiled
programs, and owns the device memory involved. The TornadoVM backend is the primary one;
the plain-Java CPU path is a first-class second. CUDA, OpenCL and Metal are *TornadoVM*
backends — device capabilities beneath one jitllm backend, not jitllm backends of
their own.

**Engine** — the tier that owns work across sequences: admission, batch composition, KV
management, prefix reuse, preemption. It sits above sessions and below the public API, and
nothing below it may depend on it.

## Words that mean more than one thing

| Term | Say instead |
| --- | --- |
| engine | *token-generation loop* for `TokenGenerationLoop`, *engine tier* for `LLMEngine` |
| batch | *batch prefill* (one sequence, many prompt tokens) or *batch decode* (many sequences, one token each) |
| slot / session | a slot is a position in the current batch; a session is a user-facing sequence |
| plan | *compiled program*, *task graph*, or `TornadoExecutionPlan` — be specific |
| model | *model file*, *model architecture*, or *loaded model* |
| state | *session state*, *KV cache*, or *invocation buffers* |
| backend | *jitllm backend* or *TornadoVM device backend* — different levels |
| tensor | `FloatTensor` is a flat float sequence, not a shaped tensor; say *buffer*, *weight tensor* or *tensor descriptor* |
