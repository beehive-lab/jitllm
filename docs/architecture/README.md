# jitllm — engine documentation

A Java-native transformer inference engine that compiles reusable Java inference
components through TornadoVM into execution plans for local accelerators. TornadoVM is the
compiler and runtime.

## Where to look

| Document | For |
| --- | --- |
| [`architecture.md`](architecture.md) | layers, dependency direction, principal abstractions, provider discovery |
| [`execution.md`](execution.md) | loading, sessions, programs, backend lowering, program caching, execution modes, sampling |
| [`api.md`](api.md) | the stable facade, options, conversations, tools, streaming, lifecycle |
| [`memory-and-concurrency.md`](memory-and-concurrency.md) | ownership, close semantics, KV storage, shared workspaces, what serializes |
| [`models-and-backends.md`](models-and-backends.md) | adding a family, dtypes and materialization, per-backend capability and limitations |
| [`verification.md`](verification.md) | gates, numerical references, the CI matrix, performance methodology, known limitations |
| [`development.md`](development.md) | JDK 21/25 builds, TornadoVM setup, adding a family or capability, release and integrations |

## Supported platforms

| | |
| --- | --- |
| JDKs | 21 and 25. Each publishes its own artifact — `jitllm:<version>-jdk21` and `-jdk25` — against its own TornadoVM line. Any other JDK fails the build with a message naming both |
| Backends | CPU (plain Java, Vector API), and through TornadoVM: CUDA, OpenCL and Metal |
| Platforms | Linux on NVIDIA (CUDA and OpenCL), macOS on Apple silicon (Metal) |
| Model formats | GGUF, in F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K and Q6_K |
| Families | Llama, Mistral, Qwen2, Qwen3, DeepSeek-R1-Distill-Qwen, Granite, Phi-3, Gemma-4, Devstral |

## The system in one paragraph

A GGUF file is recognized by a registered provider, which loads it into an immutable
**model**: configuration, weights, tokenizer and chat format. A **session** opened from
that model owns one sequence — its position, its KV lease, its sampler, its conversation.
Generating turns the family's architecture description into a backend-neutral **inference
program**, which a **backend** compiles once into a **compiled program** with a fixed
device workspace, and then invokes many times. Invocations move values into persistent
control arrays; they never rebind buffers. KV storage lives in a pooled block store that
sessions lease from, so several sessions can share one compiled program and one device copy
of the weights. Above sessions, an optional **engine** tier batches requests across
sequences. The public facade exposes none of this: no TornadoVM type, no GGUF type and no
CLI type appears in any public signature.

## Constraints that are not negotiable

These are enforced by tests, not by convention, and each one exists because breaking it
produced a real failure:

- **Compile once, execute many.** A compiled program is never rebuilt per token, and an
  invocation allocates nothing.
- **Fixed binding identity.** Every device array bound into a captured graph keeps its
  identity for the program's life. Captured-graph replay bakes addresses, and re-pointing a
  buffer produces wrong output rather than an error.
- **Backends own lowering and fusion.** Nothing above the backend SPI expresses kernel
  choice, and nothing above it can observe one except through metrics.
- **Materialization is a backend concern.** Where a quantized weight is decoded is decided
  by the backend that executes it, not by the model or the kernel.
- **No silent fallback.** An unsupported device, lowering or storage request is refused,
  never quietly substituted.
- **Dependencies point downward.** TornadoVM types stay inside the Tornado backend, and
  nothing below the engine depends on the engine.
- **Numerical claims are scored against the CPU.** A GPU-versus-GPU comparison cannot see a
  defect that moves the whole GPU, and a process that exits 0 at a normal throughput has
  proved nothing.
- **Bounds are never widened and goldens are never regenerated to make a failure pass.**
