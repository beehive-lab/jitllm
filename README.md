# jllm — LLM inference & serving for the JVM, on any GPU

[![build JDK21](https://github.com/beehive-lab/jllm/actions/workflows/build-and-run.yml/badge.svg)](https://github.com/beehive-lab/jllm/actions/workflows/build-and-run.yml)
[![Maven Central](https://img.shields.io/maven-central/v/io.github.beehive-lab/jllm?&logo=apache-maven&color=blue)](https://central.sonatype.com/artifact/io.github.beehive-lab/jllm)
![Java 21](https://img.shields.io/badge/java-21-blue?logo=openjdk)
![Java 25](https://img.shields.io/badge/java-25-yellow?logo=openjdk)
[![LangChain4j](https://img.shields.io/badge/LangChain4j-1.7.1+-purple?&logo=link&logoColor=white)](https://docs.langchain4j.dev/)
![NVIDIA](https://img.shields.io/badge/CUDA-supported-76B900?logo=nvidia)
![OpenCL](https://img.shields.io/badge/OpenCL-supported-blue?logo=khronos)
![Apple](https://img.shields.io/badge/Metal-Apple%20Silicon-black?logo=apple)
[![Docker](https://img.shields.io/badge/Docker-OpenCL%20%7C%20PTX-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/r/beehivelab/gpullama3.java-nvidia-openjdk-opencl)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/beehive-lab/jllm)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

-----------

<table style="border: none;">
<tr style="border: none;">
<td style="width: 40%; vertical-align: middle; border: none;">
<img src="docs/ll.gif" >
</td>
<td style="vertical-align: middle; padding-left: 20px; border: none;">

### Think vLLM — but pure Java, and it runs on **any** GPU.

**jllm** is a JVM-native LLM inference and serving engine. You write and ship plain Java; [**TornadoVM**](https://github.com/beehive-lab/TornadoVM) JIT-compiles the hot transformer kernels to **CUDA, OpenCL, or Apple Metal** at runtime — no JNI glue, no second toolchain, no native rebuild per GPU.

One `.jar` runs the same model on **NVIDIA, Intel, AMD, and Apple Silicon**, from a laptop to an RTX 5090.

Serve it behind an **OpenAI-compatible API**, embed it in **LangChain4j** or **Quarkus**, or run it from the CLI in one line.

</td>
</tr>
</table>

Builds on [Llama3.java](https://github.com/mukel/llama3.java) by [Alfonso² Peterssen](https://github.com/mukel). Earlier Llama2 work: [llama2.tornadovm](https://github.com/mikepapadim/llama2.tornadovm.java).

-----------

## Why jllm

- 🟦 **Pure Java, all the way down.** Transformer kernels are written in Java and accelerated by TornadoVM — no CUDA C, no hand-written JNI. Debug and build with the toolchain you already have.
- 🌍 **Write once, run on any GPU.** NVIDIA (CUDA), Intel & AMD (OpenCL), Apple Silicon (Metal). Backend is auto-detected from your TornadoVM SDK — switch with a flag, not a rebuild.
- 🔌 **Drop-in for the Java AI stack.** Official [LangChain4j](https://docs.langchain4j.dev/integrations/language-models/gpullama3-java) provider (since v1.7.1) and [Quarkus](https://docs.quarkiverse.io/quarkus-langchain4j/dev/gpullama3-chat-model.html) inference engine.
- ⚡ **Built to serve.** OpenAI-compatible HTTP server, llama-bench-style benchmarking, and tensor-core (MMA) batch prefill (see [Serving](#-serving-openai-compatible-preview)).
- 📦 **Many models, one runtime.** Llama 3, Mistral, Qwen 2.5 / Qwen 3, Phi-3, IBM Granite 3.3 / 4.0, DeepSeek-R1-Distill — all in GGUF.

-----------

## ⏱️ Quickstart (60 seconds)

```bash
# 1. Install a TornadoVM SDK (bundles the GPU runtime)
curl -s "https://get.sdkman.io" | bash && source "$HOME/.sdkman/bin/sdkman-init.sh"
sdk install tornadovm
tornado --devices        # confirm your GPU is listed

# 2. Run a model on the GPU — no build required, via JBang
jbang gpullama3@beehive-lab -m beehive-llama-3.2-1b-instruct-fp16.gguf -p "Explain GPU acceleration in one sentence."
```

Grab a ready-to-run model from the [Hugging Face collections](#-model-collections) below.

-----------

## 🧩 Serving: OpenAI-compatible (preview)

jllm is growing into a **serving engine** — the vLLM-style path for the JVM:

- 🌐 **OpenAI-compatible server** — `jllm serve` exposes `/v1/chat/completions` and `/v1/completions` with streaming and zero external dependencies. `/v1/models` reports the served context length, so clients size their prompts instead of guessing. Point any OpenAI client at `localhost`.
- 🎯 **Tensor-core (MMA) batch prefill** on the CUDA backend, FP16 & Q8_0 — `--with-prefill-decode --batch-prefill-size N`.
- 📈 **llama-bench-style benchmarking** — `jllm --bench` reports a pp/tg matrix with avg±stddev in md/csv/json/jsonl/sql. See [Running the CLI](#-running-the-cli) for the flag.
- 🧮 **On-device greedy sampling** *(landing next)* — argmax on the GPU keeps logits device-side, cutting device→host traffic by ~500× per token. ([PR #134](https://github.com/beehive-lab/jllm/pull/134))
- 📚 **Static batched decode** *(landing next)* — B independent sequences per step for up to **41× aggregate throughput** (Llama & Qwen3). ([PR #129](https://github.com/beehive-lab/jllm/pull/129))

-----------

## <img src="https://github.com/user-attachments/assets/51b76554-0b01-4e18-a567-600901ab8c5f" alt="LangChain4j" height="30" style="vertical-align: middle; margin-right: 8px;"> LangChain4j & Quarkus

Since **LangChain4j v1.7.1**, `jllm` is an officially supported **model provider** — no glue code, GPU-accelerated out of the box.

```java
GPULlama3ChatModel model = GPULlama3ChatModel.builder()
        .modelPath(modelPath)
        .temperature(0.9)      // more creative
        .topP(0.9)             // more variety
        .maxTokens(2048)
        .onGPU(Boolean.TRUE)   // false → lightweight CPU llama3.java
        .build();
```

📖 [LangChain4j docs](https://docs.langchain4j.dev/) · 🚀 [Agentic workflow demo](https://github.com/mikepapadim/devoxx25-demo-gpullama3-langchain4j/tree/main)

### 📦 Maven

<!-- DEPENDENCY-SNIPPETS:START -->
**JDK 21** (`jdk21` profile, auto-activates for JDK `[21,25)`):
```xml
<dependency>
    <groupId>io.github.beehive-lab</groupId>
    <artifactId>jllm</artifactId>
    <version>1.0.0-jdk21</version>
</dependency>
```

**JDK 25** (`jdk25` profile, auto-activates for JDK `[25.0.2,)`):
```xml
<dependency>
    <groupId>io.github.beehive-lab</groupId>
    <artifactId>jllm</artifactId>
    <version>1.0.0-jdk25</version>
</dependency>
```

### 📦 Gradle

**JDK 21**:
```groovy
implementation 'io.github.beehive-lab:jllm:1.0.0-jdk21'
```

**JDK 25**:
```groovy
implementation 'io.github.beehive-lab:jllm:1.0.0-jdk25'
```
<!-- DEPENDENCY-SNIPPETS:END -->

-----------

#### **[Interactive mode]** — RTX 5090, with `nvtop` tracking GPU utilization and memory

![Demo](docs/inter-output.gif)

-----------

## 🛠️ Install & build

### Prerequisites

- **Java 21 or 25** — required for the Vector API & TornadoVM. Each line has its own artifact (`-jdk21` / `-jdk25`); both launchers work on either, while `jllm4j` itself needs Java 25 to run.
- **[TornadoVM](https://github.com/beehive-lab/TornadoVM)** with an OpenCL, CUDA, or Metal backend. `jllm`/`jllm4j` auto-detect whichever backend your installed SDK was built with.
- **GCC/G++ 13+** — to build TornadoVM's native components.

### TornadoVM: released SDK for running, `develop` for building this branch

A **released** TornadoVM SDK — from the [official website](https://www.tornadovm.org/downloads) or
[SDKMAN!](https://sdkman.io/sdks/tornadovm/) (`sdk install tornadovm`) — runs the published jllm
artifacts and the JBang catalog. **Building this branch from source is different:** its kernels
use TornadoVM APIs that exist only on TornadoVM's `develop` branch (in-kernel reads of MMA
accumulators, byte-offset int8 fragment loads, `cp.async` staging), so the build depends on
`<version>-jdk21-dev` / `-jdk22plus-dev` artifacts that are not on Maven Central and do not come
with SDKMAN. The native prefill path also needs `tornado-cublas` and `tornado-cudnn`,
which are built from the same revision and are not available on Maven Central.
`scripts/tornadovm-dev.sh` prepares all of them the same way CI does.

### Build from source (development)

```bash
git clone https://github.com/beehive-lab/jllm.git && cd jllm

# 1. Fresh checkout: resolve upstream TornadoVM develop, build the SDK for your backend and JDK
#    (cuda | opencl | metal; auto-detected if omitted), install its Maven artifacts, record what
#    was built. Nothing here touches an SDK you installed yourself.
scripts/tornadovm-dev.sh setup --backend cuda --jdk 21

# 2. Build jllm against exactly that TornadoVM (./mvnw with -Dtornadovm.version=<what was built>)
scripts/tornadovm-dev.sh build clean install -DskipTests

# 3. Run with the matching SDK
eval "$(scripts/tornadovm-dev.sh env)"        # exports TORNADOVM_HOME and PATH for this shell
./jllm --gpu --model model.gguf --prompt "..."
```

| task | command |
|---|---|
| Rebuild without touching TornadoVM | `scripts/tornadovm-dev.sh build clean package -DskipTests` (reuses the prepared installation) |
| Build against one specific installation | `scripts/tornadovm-dev.sh build --install ~/.jllm/tornadovm/cuda-jdk21/<commit>-r<recipe> …` |
| Remove old installations | `scripts/tornadovm-dev.sh prune --yes` (explicit; never automatic) |
| Advance to the latest `develop` | `scripts/tornadovm-dev.sh refresh --backend cuda --jdk 21` |
| Reproduce an exact revision | `scripts/tornadovm-dev.sh setup --ref <40-hex commit> --backend cuda --jdk 21` |
| What is prepared | `scripts/tornadovm-dev.sh status` (revision, artifact version, JDK, backend, SDK path) |
| Build a jllm release tag | check out the tag, then `./mvnw -P release -Dtornadovm.release.version=<X.Y.Z> clean package` — a release depends on a published TornadoVM release and refuses `-dev` coordinates |

`scripts/tornadovm-dev.sh build` is the reliable build entry point: it reuses the prepared
installation, passes the exact artifact version that installation produced and that
installation's own Maven repository to `./mvnw`, and never fetches or rebuilds TornadoVM on its
own (launching `jllm` never does either). Installations live under
`~/.jllm/tornadovm/<backend>-jdk<N>/<commit>-r<recipe>/` — the checkout, its SDK, its Maven
repository (`m2/`) and a `provenance.json` — and are immutable; `current` is only a convenience
pointer to the last one prepared, and nothing is deleted unless you run
`scripts/tornadovm-dev.sh prune --yes` (which removes every installation on that line except
`current`; do not run it while a build or an inference process is using an older one).

**Plain `./mvnw` does not see these artifacts.** TornadoVM's develop artifacts carry the same
coordinates for every commit (`6.1.1-jdk21-dev`), so they are kept in each installation's own
repository rather than in `~/.m2`, and the POM's development default only names a version, it does
not track develop. A plain `./mvnw` therefore needs `-Dmaven.repo.local=<installation>/m2
-Dtornadovm.version=<its artifact_version>` (both printed by `scripts/tornadovm-dev.sh status`),
which is exactly what `scripts/tornadovm-dev.sh build` adds; without them it fails to resolve
`tornado-api:…-dev`, and that failure is the signal to use the helper.

-----------

## ▶️ Running the CLI

Use the `jllm` script with `--gpu`. The backend (OpenCL, CUDA, or Metal) is auto-detected from
your installed TornadoVM SDK (`TORNADOVM_HOME/etc/tornado.backend`) — no need to select it manually. If your
SDK was built with more than one backend, force one with `--opencl`, `--cuda` (NVIDIA), or `--metal`
(Apple Silicon); forcing a backend that isn't part of the installed SDK errors out.

```bash
# Basic GPU inference — backend auto-detected
./jllm --gpu --verbose \
  --model beehive-llama-3.2-1b-instruct-fp16.gguf \
  --prompt "Explain the benefits of GPU acceleration."

# Force a specific backend (only needed for multi-backend SDKs)
./jllm --gpu --cuda \
  --model beehive-llama-3.2-1b-instruct-fp16.gguf \
  --prompt "Explain the benefits of GPU acceleration."
```

Swap in any tested model — e.g. `beehive-llama-3.2-3b-instruct-fp16.gguf` or `...-8b-...`.

### `jllm4j` — zero-dependency Java 25 script

Same backend auto-detection as `jllm`. A single-file Java 25 launcher that replaces the Python
script (needs `java 25+` on your PATH):

```bash
./jllm4j --gpu --verbose-init --metal \
  --model Mistral-7B-Instruct-v0.3.Q8_0.gguf --prompt "what is java"
```

### 🚀 JBang — run without building

Script-like startup à la [Jlama](https://github.com/tjake/Jlama), powered by [JBang](https://www.jbang.dev/):

```bash
curl -Ls https://sh.jbang.dev | bash -s - app setup

# From the catalog
jbang gpullama3@beehive-lab -m model.gguf -p "Tell me a joke"
jbang app install gpullama3@beehive-lab && gpullama3 -m model.gguf -p "Hello!"
```

> The `gpullama3@beehive-lab` alias is the pre-rename catalog name and still resolves.
> See [`POST_MOVE_ACTIONS.md`](POST_MOVE_ACTIONS.md) for the migration status.

### 🐳 Docker

Fully containerized GPU inference via pre-built images ([docker-gpullama3.java](https://github.com/beehive-lab/docker-gpullama3.java)):

| Backend | Image |
|---------|-------|
| **OpenCL** | [`beehivelab/gpullama3.java-nvidia-openjdk-opencl`](https://hub.docker.com/r/beehivelab/gpullama3.java-nvidia-openjdk-opencl) |
| **PTX (CUDA)** | [`beehivelab/gpullama3.java-nvidia-openjdk-ptx`](https://hub.docker.com/r/beehivelab/gpullama3.java-nvidia-openjdk-ptx) |

```bash
docker run --rm -it --gpus all -v "$PWD":/data \
  beehivelab/gpullama3.java-nvidia-openjdk-opencl \
  /gpullama3/GPULlama3.java/llama-tornado \
  --gpu --verbose \
  --model /data/Llama-3.2-1B-Instruct.FP16.gguf --prompt "Tell me a joke"
```

-----------

## 🤗 Model collections

GGUF models, ready to download:

| Family | Collection |
|--------|-----------|
| Llama 3.2 | [llama3-gpullama3java](https://huggingface.co/collections/beehive-lab/llama3-gpullama3java) |
| IBM Granite 4.0 | [granite-40-language-models](https://huggingface.co/collections/beehive-lab/granite-40-language-models-gpullama3java) |
| IBM Granite 3.3 | [granite-33-language-models](https://huggingface.co/collections/beehive-lab/granite-33-language-models-gpullama3java) |
| Qwen 2.5 | [qwen-25-gpullama3java](https://huggingface.co/collections/beehive-lab/qwen-25-gpullama3java) |
| Qwen 3 | [qwen-3-gpullama3java](https://huggingface.co/collections/beehive-lab/qwen-3-gpullama3java) |
| Phi-3 | [phi-3-gpullama3java](https://huggingface.co/collections/beehive-lab/phi-3-gpullama3java) |
| Mistral | [mistral-gpullama3java](https://huggingface.co/collections/beehive-lab/mistral-gpullama3java) |
| DeepSeek-R1-Distill-Qwen | [deepseek-r1-distill-qwen](https://huggingface.co/collections/beehive-lab/deepseek-r1-distill-qwen-gpullama3java) |

Formats: GGUF · FP16 (full), Q8_0 & Q4_0 (partial).

-----------

## 💾 GPU memory

Default device allocation is **14GB**. Larger models need more — raise it with `--gpu-memory`:

| Model size | Recommended | Flag |
|------------|-------------|------|
| 1B  | 14GB (default) | — |
| 3–7B | 15GB+ | `--gpu-memory 15GB` |
| 8B+ | 20GB+ | `--gpu-memory 20GB` |

```bash
./jllm --gpu --model beehive-llama-3.2-3b-instruct-fp16.gguf \
  --prompt "Tell me a joke" --gpu-memory 15GB
```

Still out of memory? Use Q4_0 instead of Q8_0, or close other GPU apps. The error to look for:

```
org.beehive.jllm.api.InsufficientDeviceMemoryException: [GPUL-MEM-001] This configuration
needs about 13826.6 MiB of device memory but the configured budget is 1024.0 MiB
(short by 12802.6 MiB).
  Dominant component: weights (per-layer) at 13313.0 MiB
  Raise -Dtornado.device.memory, reduce the context length, or select a smaller quantization.
```

It is followed by a per-component device memory plan, so you can see what is actually
consuming the budget before changing anything.

-----------

## Run configuration at startup

With `-v` / `--verbose`, the CLI prints an aligned summary to **stderr**, after preparing the session and before
printing generated text. It shows the model filename, parameter count and file size,
GGUF quantization alongside loaded weight types, device, TornadoVM version on GPU runs, execution mode, context
capacity, batched prefill chunk width, KV-cache precision, MMA/tensor-core selection,
native libraries, CUDA graphs, staged transfers, GPU allocation budget, and sampling settings.
GPU memory estimates split weights, KV cache, and workspace (including staging and
control buffers), with a total. These predict allocation-budget charges, not physical
VRAM use; conservative predictions are labeled. CPU runs instead show common-pool worker
parallelism plus the caller and the configured tensor/Q4 Vector API settings.
Tensor-core and native-library details describe the selected prefill path; native
libraries choose their own kernel algorithms.

Startup timings separate model loading, plan construction, TornadoVM JIT precompilation,
and initial device setup. Device setup includes uploads and execution (including CUDA
graph capture when enabled), so it is **not** a pure transfer measurement. “Ready to
generate” measures elapsed startup time through session preparation. Prefill and decode
performance still appear after generation.

Without `--verbose`, startup diagnostics are hidden and session preparation stays lazy;
errors, warnings, generated text, and final performance metrics remain visible.
Verbose output includes SDK location, dimensions/head counts, training context,
execution path, and memory-estimate assumptions. Startup timings are printed once;
the ending performance block contains only request metrics.

`--verbose-init` remains a hidden deprecated alias for `--verbose`.
For direct Java launches, use `-Djllm.verbose=true`. The legacy
`-Djllm.EnableTimingForTornadoVMInit=true` setting still enables the report and, in
addition, the older per-stage initialization log lines; `--verbose` no longer sets it.
The full Java command is printed by `--show-command`, not by `--verbose`.
The summary is CLI-only; library callers can explicitly use `GenerationSession.prepare()`
to prepare a session and obtain its execution settings without advancing its position.

-----------

## 🔧 Embed in your own tools

`--show-command` prints the exact Java + JVM invocation used under the hood, so you can replicate it in IntelliJ, Maven, Gradle, or any launcher:

```bash
jllm --gpu --model beehive-llama-3.2-1b-instruct-fp16.gguf \
  --prompt "tell me a joke" --show-command
```

Each command has focused help:

```bash
./jllm --help
./jllm run --help
./jllm chat --help
./jllm serve --help
./jllm bench --help
```

| Command | Purpose | Key options |
| --- | --- | --- |
| `run` | Generate one response, then exit | `--prompt`, `--system-prompt`, `--max-new-tokens` |
| `chat` | Terminal conversation using one persistent session | `--system-prompt`, `--max-new-tokens` per turn |
| `serve` | OpenAI-compatible HTTP API | `--host`, `--port`; experimental `--continuous-batching` |
| `bench` | Repeated prefill/decode workloads | `--pp`, `--tg`, `--depth`, `--repetitions`, `--output` |

Model, GPU/backend selection, KV precision, execution policy, and `-v`/`--verbose`
are shared. `--ctx-size` (`-c`, also `--ctx`/`--context-length`) sets context capacity
for run/chat/serve; `--max-new-tokens` separately limits generated tokens in run/chat.
The default context is 512 tokens for run/chat and the model's own for serve (a larger
request is clamped to it); the default generation limit is the context capacity.
Benchmark context is derived from its workload sizes and depths.

```bash
./jllm run -m model.gguf --gpu -c 4096 --max-new-tokens 128 --prompt "Explain SIMD."
./jllm chat -m model.gguf --gpu -c 4096 --max-new-tokens 128
./jllm serve -m model.gguf --gpu -c 4096 --host 127.0.0.1 --port 8080 -v
./jllm bench -m model.gguf --gpu --pp 128,512 --tg 64 --depth 0,4096 --repetitions 3 --output json
```

**Key/value cache precision.** The KV cache is stored in **FP16 by default** (accumulation
stays FP32). `--fp32-kv-cache` selects FP32, the compatibility and numerical-reference
choice. A configuration whose kernels do not implement the FP16 cache is refused before
the model's cache or plan is built, naming the combination and pointing at
`--fp32-kv-cache`; it never falls back to FP32 silently. The supported set is tracked in
[docs/architecture/kv-cache-support.md](docs/architecture/kv-cache-support.md). The old
`--fp16-kv-cache` flag was removed and is refused with this migration.

Existing flag-based invocations remain supported: default/`--instruct` → `run`,
`--interactive`/`--chat`/`-i` → `chat`, `--server` → `serve`, and `--bench` → `bench`.
Conflicting modes and options for another command are rejected. `--max-tokens`/`-n`
remains a deprecated **context-capacity** alias; it has not been repurposed as an output
limit. HTTP `max_tokens` keeps its existing generated-token meaning. Legacy
`--bench-args="..."` still accepts benchmark arguments, including quoted values.

Serving binds to loopback by default and prints its actual address and port when ready.
Use `--host 0.0.0.0` explicitly for all IPv4 interfaces. Clients submit conversation
history in each HTTP request; `chat` retains terminal conversation history locally.

**Experimental: continuous batching.** `serve --continuous-batching SLOTS` decodes up to
`SLOTS` HTTP requests together in one batch instead of one at a time, with
`--max-queued-requests` and `--prefix-cache-entries` as its options (listed under
*Experimental* in `jllm serve --help`). It prints a warning when enabled and currently supports
CUDA tensor-core devices, FP16 Llama/Qwen3 weights, an FP32 KV cache (`--fp32-kv-cache`,
required) and greedy (`temperature=0`) requests only. Prefill chunking and CUDA graphs are
rejected in that mode because this executor does not implement them. JIT/device setup remains
lazy there and is labeled accordingly in verbose output; its memory estimate is unavailable.
Prefix caching requires continuous batching and is disabled by default. Not to be confused with
`--batch-prefill-size N`, which counts **prompt tokens per chunk** of one request.

`serve -v` and `bench -v` use the same startup report as terminal generation. Server
request sampling and benchmark token workloads are labeled appropriately; reports stay
on stderr so HTTP responses and benchmark JSON/CSV stay separate.


```bash
# Peek at what TornadoVM is doing
./jllm --gpu --model model.gguf --prompt "..." --print-kernel      # generated GPU kernel
./jllm --gpu --model model.gguf --prompt "..." --print-bytecodes   # TornadoVM bytecodes
./jllm --gpu --model model.gguf --prompt "..." --debug --full-dump # everything
```

-----------

## 🗺️ Features & roadmap

- ✅ **GGUF models** — full FP16, partial Q8_0 / Q4_0.
- ✅ **Chat, instruction, and interactive** modes (`--interactive`, `--instruct`).
- ✅ **Automatic backend detection** — `jllm`/`jllm4j` detect and use whichever backend (OpenCL, CUDA, or Metal) your installed TornadoVM SDK was built with; override with `--opencl`/`--cuda`/`--metal`.
- ✅ **Cross-platform**: NVIDIA (OpenCL · CUDA), Intel (OpenCL), Apple (OpenCL · Metal).
- ✅ **Serving** — OpenAI-compatible API, llama-bench-style benchmarking, tensor-core (MMA) batch prefill.
- ✅ **Native batched prefill** (Qwen3 FP16, CUDA) — cuBLAS projections and a fused cuDNN first-chunk attention, selected automatically with no flag.
- ✅ **Faster CUDA decode** (Qwen3 FP16) — grouped decode graphs, warp-butterfly matrix-vector reductions and a lane-cooperative attention kernel, all selected by device capability with no flag.
- 🧩 **Coming next** — static batched decode, on-device sampling (preview; see [Serving](#-serving-openai-compatible-preview)).

📄 [Transformer optimizations in TornadoVM](docs/TORNADOVM_TRANSFORMER_OPTIMIZATIONS.md) · 🧭 [Project roadmap](docs/jllm-roadmap.md)

-----------

## 🙏 Acknowledgments

Partially funded by EU Horizon Europe & UKRI grants (most recent first):
[AERO 101092850](https://aero-project.eu/) · [P2CODE 101093069](https://p2code-project.eu/) · [ENCRYPT 101070670](https://encrypt-project.eu) · [TANGO 101070052](https://tango-project.eu).

## License

[MIT](LICENSE)
