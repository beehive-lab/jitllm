# jitLLM: LLM inference & serving for the JVM, on any GPU

[![build JDK21](https://github.com/beehive-lab/jitllm/actions/workflows/build-and-run.yml/badge.svg)](https://github.com/beehive-lab/jitllm/actions/workflows/build-and-run.yml)
[![Maven Central](https://img.shields.io/maven-central/v/io.github.beehive-lab/jitllm?&logo=apache-maven&color=blue)](https://central.sonatype.com/artifact/io.github.beehive-lab/jitllm)
![Java 21](https://img.shields.io/badge/java-21-blue?logo=openjdk)
![Java 25](https://img.shields.io/badge/java-25-yellow?logo=openjdk)
[![LangChain4j](https://img.shields.io/badge/LangChain4j-1.7.1+-purple?&logo=link&logoColor=white)](https://docs.langchain4j.dev/)
![NVIDIA](https://img.shields.io/badge/CUDA-supported-76B900?logo=nvidia)
![OpenCL](https://img.shields.io/badge/OpenCL-supported-blue?logo=khronos)
![Apple](https://img.shields.io/badge/Metal-Apple%20Silicon-black?logo=apple)
[![Docker](https://img.shields.io/badge/Docker-OpenCL-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/r/beehivelab/gpullama3.java-nvidia-openjdk-opencl)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/beehive-lab/jitllm)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

-----------

<table style="border: none;">
<tr style="border: none;">
<td style="width: 40%; vertical-align: middle; border: none;">
<img src="docs/ll.gif" >
</td>
<td style="vertical-align: middle; padding-left: 20px; border: none;">

### Think vLLM — but pure Java, and it runs on **any** GPU.

**jitLLM** is a JVM-native LLM inference and serving engine. You write and ship plain Java; [**TornadoVM**](https://github.com/beehive-lab/TornadoVM) JIT-compiles the hot transformer kernels to **CUDA, OpenCL, or Apple Metal** at runtime — no JNI glue, no second toolchain, no native rebuild per GPU.

One `.jar` runs the same model on **NVIDIA, Intel, AMD, and Apple Silicon**, from a laptop to an RTX 5090.

Serve it behind an **OpenAI-compatible API**, embed it in **LangChain4j** or **Quarkus**, or run it from the CLI in one line.

</td>
</tr>
</table>


-----------

## Why jitLLM

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
jbang jitllm@beehive-lab -m beehive-llama-3.2-1b-instruct-fp16.gguf -p "Explain GPU acceleration in one sentence."
```

Grab a ready-to-run model from the [Hugging Face collections](#-model-collections) below.

-----------

## 🧩 Serving: OpenAI-compatible (preview)

jitLLM is growing into a **serving engine** — the vLLM-style path for the JVM:

- 🌐 **OpenAI-compatible server** — `jitllm serve` exposes `/v1/chat/completions` and `/v1/completions` with streaming and zero external dependencies. `/v1/models` reports the served context length, so clients size their prompts instead of guessing. Point any OpenAI client at `localhost`.
- 🎯 **Tensor-core batch prefill on CUDA for every model family** (F16, Q8_0, Q4_0, Q4_K) — on by default with `--gpu`, cuBLAS projections included where cuBLAS is installed. Llama-3.2-1B Q8_0 prompt processing 4.9k → 29k tok/s on an A100, past llama.cpp; Qwen3-0.6B F16 54k tok/s with cuDNN attention. See [Fast prompt processing on CUDA](#fast-prompt-processing-on-cuda).
- 🍎 **SIMD-group matrix batch prefill on Apple Metal**, on by default for every supported family — e.g. Llama-3.2-1B F16 prompt processing 40 → 2372 tok/s, Qwen3-0.6B F16 4171 tok/s on an M4 Pro.
- 📈 **llama-bench-style benchmarking** — `jitllm --bench` reports a pp/tg matrix with avg±stddev in md/csv/json/jsonl/sql. See [Running the CLI](#-running-the-cli) for the flag.
- 🧮 **On-device greedy sampling** *(landing next)* — argmax on the GPU keeps logits device-side, cutting device→host traffic by ~500× per token. ([PR #134](https://github.com/beehive-lab/jitllm/pull/134))
- 📚 **Static batched decode** *(landing next)* — B independent sequences per step for up to **41× aggregate throughput** (Llama & Qwen3). ([PR #129](https://github.com/beehive-lab/jitllm/pull/129))

-----------

## <img src="https://github.com/user-attachments/assets/51b76554-0b01-4e18-a567-600901ab8c5f" alt="LangChain4j" height="30" style="vertical-align: middle; margin-right: 8px;"> LangChain4j & Quarkus

Since **LangChain4j v1.7.1**, `jitllm` is an officially supported **model provider** — no glue code, GPU-accelerated out of the box.

```java
JitLLMChatModel model = JitLLMChatModel.builder()
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
    <artifactId>jitllm</artifactId>
    <version>1.0.2-jdk21</version>
</dependency>
```

**JDK 22+** (`jdk22plus` profile, auto-activates for JDK `[22,)`):
```xml
<dependency>
    <groupId>io.github.beehive-lab</groupId>
    <artifactId>jitllm</artifactId>
    <version>1.0.2-jdk22plus</version>
</dependency>
```

### 📦 Gradle

**JDK 21**:
```groovy
implementation 'io.github.beehive-lab:jitllm:1.0.2-jdk21'
```

**JDK 22+**:
```groovy
implementation 'io.github.beehive-lab:jitllm:1.0.2-jdk22plus'
```
<!-- DEPENDENCY-SNIPPETS:END -->

The artifact is a library jar: it carries jitllm's own classes and nothing else. TornadoVM and
its dependencies come from the TornadoVM SDK at runtime, so start the JVM with
`@$TORNADOVM_HOME/tornado-argfile --add-modules jdk.incubator.vector`. Code that uses TornadoVM
types directly declares `tornado-api` itself, with `provided` scope.

-----------

#### **[Interactive mode]** — RTX 5090, with `nvtop` tracking GPU utilization and memory

![Demo](docs/inter-output.gif)

-----------

## 🛠️ Install & build

### Prerequisites

- **Java 21 or 22+** — required for the Vector API & TornadoVM. Each line has its own artifact (`-jdk21` / `-jdk22plus`), matching TornadoVM's; both launchers work on either, while `jitllm4j` itself needs Java 25 to run.
- **[TornadoVM](https://github.com/beehive-lab/TornadoVM)** with an OpenCL, CUDA, or Metal backend. `jitllm`/`jitllm4j` auto-detect whichever backend your installed SDK was built with.
- **GCC/G++ 13+** — to build TornadoVM's native components.

### TornadoVM

jitllm compiles against, and runs on, the TornadoVM SDK that `TORNADOVM_HOME` points at. Any SDK
from **TornadoVM 7.3** on works: a released one, from the
[official website](https://www.tornadovm.org/downloads) or [SDKMAN!](https://sdkman.io/sdks/tornadovm/)
(`sdk install tornadovm`), or a local build of TornadoVM `develop`. 7.3 is the first release with
the half-precision SIMD-group matrix API the Metal batched prefill compiles against. Its JDK line
must match the JDK you build with: a `-jdk21` SDK for JDK 21, a `-jdk22plus` SDK for JDK 22 and
newer. The build checks both that `TORNADOVM_HOME` is set and that the line matches.

### Build from source

```bash
git clone https://github.com/beehive-lab/jitllm.git && cd jitllm

export TORNADOVM_HOME=/path/to/tornadovm-sdk     # e.g. ~/.sdkman/candidates/tornadovm/current
export PATH="$TORNADOVM_HOME/bin:$PATH"
mvn clean package -DskipTests                    # or ./mvnw; drop -DskipTests to run the unit tests

./jitllm --gpu --model model.gguf --prompt "..."
```

Nothing is resolved from a Maven repository for TornadoVM, so there is no version to pass: switch
SDKs by changing `TORNADOVM_HOME`, and `clean` when you do.

| task | command |
|---|---|
| IDE support (IntelliJ IDEA, VS Code, Eclipse) | enable the `ide` Maven profile; it adds the SDK's jars as dependencies. For an SDK other than the release in `pom.xml` (`tornadovm.release.version`), run `scripts/ide-setup.sh` first (and again after switching `TORNADOVM_HOME`): it records the SDK's jar version in `.mvn/maven.config` |
| Build against TornadoVM `develop` | `scripts/tornadovm-dev.sh setup --backend cuda --jdk 21`, then `eval "$(scripts/tornadovm-dev.sh env)"` (exports `TORNADOVM_HOME`) and build as above |
| Advance to the latest `develop` | `scripts/tornadovm-dev.sh refresh --backend cuda --jdk 21` |
| Reproduce an exact `develop` revision | `scripts/tornadovm-dev.sh setup --ref <40-hex commit> --backend cuda --jdk 21` |
| What is prepared | `scripts/tornadovm-dev.sh status` |
| Remove old `develop` installations | `scripts/tornadovm-dev.sh prune --yes` (explicit; never automatic) |
| Build a jitllm release | `./mvnw -P release -Dtornadovm.release.version=<X.Y.Z> clean package` — a release depends on a published TornadoVM release from Maven Central, not on `TORNADOVM_HOME` |

`scripts/tornadovm-dev.sh` is what CI uses to build TornadoVM `develop`. Installations live under
`~/.jitllm/tornadovm/<backend>-jdk<N>/<commit>-r<recipe>/` and are immutable; `current` is only a
convenience pointer to the last one prepared, and nothing is deleted unless you run
`scripts/tornadovm-dev.sh prune --yes` (which removes every installation on that line except
`current`; do not run it while a build or an inference process is using an older one).

-----------

## ▶️ Running the CLI

Use the `jitllm` script with `--gpu`. The backend (OpenCL, CUDA, or Metal) is auto-detected from
your installed TornadoVM SDK (`TORNADOVM_HOME/etc/tornado.backend`) — no need to select it manually. If your
SDK was built with more than one backend, force one with `--opencl`, `--cuda` (NVIDIA), or `--metal`
(Apple Silicon); forcing a backend that isn't part of the installed SDK errors out.

```bash
# Basic GPU inference — backend auto-detected
./jitllm --gpu --verbose \
  --model beehive-llama-3.2-1b-instruct-fp16.gguf \
  --prompt "Explain the benefits of GPU acceleration."

# Force a specific backend (only needed for multi-backend SDKs)
./jitllm --gpu --cuda \
  --model beehive-llama-3.2-1b-instruct-fp16.gguf \
  --prompt "Explain the benefits of GPU acceleration."
```

Swap in any tested model — e.g. `beehive-llama-3.2-3b-instruct-fp16.gguf` or `...-8b-...`.

GPU runs prefill the prompt in batches and use cuBLAS/cuDNN (CUDA) or MLX (Metal) where they are
installed; `--no-batch-prefill` and `--no-native-libraries` turn these off. See
[Fast prompt processing on CUDA](#fast-prompt-processing-on-cuda).

### `jitllm4j` — zero-dependency Java 25 script

Same backend auto-detection as `jitllm`. A single-file Java 25 launcher that replaces the Python
script (needs `java 25+` on your PATH):

```bash
./jitllm4j --gpu --verbose-init --metal \
  --model Mistral-7B-Instruct-v0.3.Q8_0.gguf --prompt "what is java"
```

### 🚀 JBang — run without building

Script-like startup à la [Jlama](https://github.com/tjake/Jlama), powered by [JBang](https://www.jbang.dev/):

```bash
curl -Ls https://sh.jbang.dev | bash -s - app setup

# From the catalog
jbang jitllm@beehive-lab -m model.gguf -p "Tell me a joke"
jbang app install jitllm@beehive-lab && jitllm -m model.gguf -p "Hello!"
```

> Runs [`JitllmCli.java`](JitllmCli.java) on the GPU through the TornadoVM SDK in
> `TORNADOVM_HOME` (JDK 22+). The pre-rename `gpullama3@beehive-lab` alias runs the same script.

-----------
## Tested Models


-----------
## 💾 GPU memory

Default device allocation is **14GB**. Larger models need more — raise it with `--gpu-memory`:

| Model size | Recommended | Flag |
|------------|-------------|------|
| 1B  | 14GB (default) | — |
| 3–7B | 15GB+ | `--gpu-memory 15GB` |
| 8B+ | 20GB+ | `--gpu-memory 20GB` |

```bash
./jitllm --gpu --model beehive-llama-3.2-3b-instruct-fp16.gguf \
  --prompt "Tell me a joke" --gpu-memory 15GB
```

Still out of memory? Use Q4_0 instead of Q8_0, or close other GPU apps. The error to look for:

```
org.beehive.jitllm.api.InsufficientDeviceMemoryException: [GPUL-MEM-001] This configuration
needs about 13826.6 MiB of device memory but the configured budget is 1024.0 MiB
(short by 12802.6 MiB).
  Dominant component: weights (per-layer) at 13313.0 MiB
  Raise -Dtornado.device.memory, reduce the context length, or select a smaller quantization.
```

It is followed by a per-component device memory plan, so you can see what is actually
consuming the budget before changing anything.

-----------

### Fast prompt processing on CUDA

With `--gpu`, the prompt is prefilled in batches by default (512 tokens per step),
and on an NVIDIA GPU with tensor cores the batched projections run in cuBLAS (MLX on Apple
silicon) where the model has a native-library path and the libraries are installed. When they
are not, the run prints a notice naming what to install and uses jitllm's own kernels:

```text
INFO: For maximum performance please install cuBLAS 12 and cuDNN 9; falling back to the JIT kernels.
```

A wider chunk and CUDA graph replay are faster still:

```bash
./jitllm run --gpu --cuda --cuda-graphs --with-prefill-decode --batch-prefill-size 512 \
  -m Llama-3.2-1B-Instruct-Q8_0.gguf --prompt "..."
```

| Flag | Effect |
| --- | --- |
| `--with-prefill-decode --batch-prefill-size N` | Prefill N tokens per step instead of the default |
| `--no-batch-prefill` | Read the prompt one token at a time |
| `--with-native-libraries` | Require cuBLAS/cuDNN or MLX, and refuse a configuration without a native path |
| `--no-native-libraries` | Keep jitllm's kernels; saves the device memory of the FP16 weight copies the native path stacks |

Batched Q8_0 projections accumulate through FP16 tensor-core GEMMs rather than the FP32
single-token path, so their logits differ slightly from single-token prefill; use
`--no-batch-prefill` when you need the single-token numbers. A model split across GPUs
(`--devices`) keeps single-token prefill unless you ask for batches.

Every model family has a batched prefill on CUDA: Llama, Mistral, Qwen2 / DeepSeek-R1-Distill-Qwen,
Qwen3, Qwen3.5 / 3.6 / 3.8, Phi-3, Granite, Gemma 4, Devstral, Qwen2-MoE and GLM-4.7-Flash.
Attention runs as tensor-core GEMMs over the FP16 key/value cache; Q8_0, Q4_0 and Q4_K weights are
decoded to FP16 for the projection GEMMs (kept decoded when the model's FP16 weights fit in 6 GB
and, under a configured `--gpu-memory` budget, one copy per session fits beside everything else;
otherwise decoded again for each chunk).
Decode stays single-token and is usually fastest that way.

Prompt processing on an NVIDIA A100 80GB (pp512, tok/s, best configuration each):

| Model | main (1.0.2) | now | llama.cpp |
| --- | --- | --- | --- |
| Llama-3.2-1B Q8_0 | 4,921 | 29,004 | 23,550 |
| Qwen3-0.6B F16 | 12,308 | 54,355 | 40,522 |
| Granite-3.2-2B Q8_0 | 58 | 13,392 | 9,814 |
| Mistral-7B Q8_0 | 35 | 5,447 | 4,600 |
| Gemma-4-31B Q8_0 | — | 1,430 | 1,021 |
| Llama-3.3-70B Q4_0 | — | 663 | 556 |

The native path needs cuBLAS 12: the TornadoVM binding loads `libcublasLt.so.12`. On a
CUDA 13 machine, put a CUDA 12 cuBLAS (for example the `nvidia-cublas-cu12` wheel's `lib`
directory) on `LD_LIBRARY_PATH`. Qwen3 F16 also runs the first prompt chunk's attention in cuDNN when
cuDNN 9 is present: build TornadoVM's `libtornado-cudnn.so` (`tornado-drivers/cudnn-jni`) against
cuDNN 9 (for example the `nvidia-cudnn-cu12` wheel) and add its `lib` directory to
`LD_LIBRARY_PATH`; with cuDNN 8.9 that attention falls back to jitllm's kernels (36k instead of
54k tok/s for Qwen3-0.6B F16). Without cuBLAS, or with `--no-native-libraries`, the same prefill
runs on jitllm's own tensor-core kernels, about 0.5–0.75× as fast.

### Fast prompt processing on Apple silicon (MLX)

On Metal, `--with-native-libraries` runs the batched prefill's attention and projections on
[Apple MLX](https://github.com/ml-explore/mlx) through TornadoVM's `tornado-mlx` library tasks;
decode stays on jitllm's kernels. It needs MLX 0.32.3's kernel library (`brew install mlx`) and an
M4-class GPU, and should run with `-Dtornado.mlx.async=true`, without which every MLX call waits
for the GPU and prefill is slower than jitllm's own kernels:

```bash
JAVA_TOOL_OPTIONS=-Dtornado.mlx.async=true ./jitllm run --gpu --with-prefill-decode \
  --batch-prefill-size 2048 --with-native-libraries -m Qwen3-0.6B-Q8_0.gguf --prompt "..."
```

Llama (F16, Q8_0, Q4_0), Mistral, Qwen2 / DeepSeek-R1-Distill-Qwen, Qwen3, Phi-3 and Granite (F16,
Q8_0) are supported. Q8_0 and Q4_0 projections are multiplied either from resident FP16 copies —
faster, at twice the Q8_0 size beside the model — or from an exact affine repack MLX decodes per
chunk; the copies are kept when they fit 6 GB and the `--gpu-memory` budget, and
`-Djitllm.prefill.decodedProjections=resident|per-chunk` pins the choice.

Prompt processing on an Apple M4 Pro (24 GB), pp2048 at a batch of 2048, FP16 key/value cache,
TornadoVM develop `3d2d9695`, `llama-bench` at `11fe021` with flash attention (tok/s, mean of 5):

| Model | jitllm kernels | MLX | MLX weights | llama.cpp |
| --- | ---: | ---: | --- | ---: |
| Qwen3-0.6B F16 | 3,257 | 4,042 | F16 | 4,233 |
| Qwen3-0.6B F16 (pp4096, batch 4096) | 2,418 | 3,467 | F16 | 3,360 |
| Qwen3-0.6B Q8_0 | 3,088 | 4,067 | FP16 copies | 4,076 |
| Qwen3-0.6B Q8_0 | | 3,834 | affine | |
| Llama-3.2-1B F16 | 2,246 | 2,613 | F16 | 2,655 |
| Llama-3.2-1B Q8_0 | 2,045 | 2,609 | FP16 copies | 2,532 |
| Llama-3.2-1B Q8_0 | | 2,398 | affine | |
| Qwen2.5-0.5B Q8_0 | 4,989 | 6,163 | FP16 copies | 5,924 |
| Phi-3-mini Q8_0 | 538 | 643 | affine | 680 |
| Phi-3-mini Q8_0 | | 705 | FP16 copies, pinned (+7.2 GB) | |

Mistral-7B Q8_0 measures 293 (jitllm) against 310 ± 69 (MLX, affine) and 372 (llama.cpp) with
`--gpu-memory 17GB`; the spread reflects memory pressure on a 24 GB machine. Under the default
14 GB budget `jitllm run` refuses the MLX load at the memory preflight (`GPUL-MEM-001`, before any
device allocation); `jitllm bench` skips the preflight and fails part-allocated instead.

-----------

### Several GPUs

A model too large for one GPU can be split by layers across several, like llama.cpp's
`--split-mode layer`. Each GPU holds a contiguous range of layers with their weights and
key/value cache, and only the hidden state moves between GPUs, once per token. For one
request at a time this adds memory, not speed. Llama-family models with Q4_0 or Q8_0 layers
are supported, and so are Qwen3.5 (`qwen35`, e.g. Qwen3.8-27B) and its mixture-of-experts
models (`qwen35moe`, e.g. Qwen3.6-35B-A3B), Gemma 4 31B and GLM-4.7-Flash (`deepseek2`).

```bash
./jitllm run --gpu --cuda --devices 0:0,0:1 --tensor-split 41,39 --gpu-memory 21GB \
  -m Llama-3.3-70B-Instruct-Q4_0.gguf -c 2048 --prompt "Why is the sky blue?"
```

- `--devices` lists TornadoVM `backend:device` indices, one pipeline stage per device.
  `--gpu-memory` applies to each device.
- `--tensor-split` gives each device's share of the layers (default: equal). Give the last
  device a little less: it also holds the output projection.
- `--split-transport nccl` (default) sends the hidden state GPU to GPU with NCCL. It needs a
  TornadoVM SDK with `tornado-nccl`, libnccl on `LD_LIBRARY_PATH`, and jitllm built with
  `-Pnccl`. `--split-transport host` copies through host memory and needs neither.

Llama-3.3-70B Q4_0 (40 GB) runs this way on two 24 GB NVIDIA A10s at about 10 tokens/s.

Requirements, the supported families and how the pipeline works are in
[docs/architecture/multi-gpu.md](docs/architecture/multi-gpu.md).

For Q4_0 models the prompt can be prefilled in chunks on every device as well: add
`--batch-prefill-size 256 --with-native-libraries` (each device runs its layers on the whole
chunk, then hands the chunk's hidden states to the next one). The chunks overlap: the first
device starts the next chunk while the next device still works on the previous one.
Qwen3.5, Gemma 4 and GLM-4.7-Flash models split the same way with `--batch-prefill-size 256`
alone: their projections run as int8 tensor-core GEMMs straight on the Q8_0 weights. A
mixture-of-experts model routes each token to its top experts on the GPU, and a prefill chunk's
tokens are grouped by expert so every expert's rows run as one GEMM.

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
For direct Java launches, use `-Djitllm.verbose=true`. The legacy
`-Djitllm.EnableTimingForTornadoVMInit=true` setting still enables the report and, in
addition, the older per-stage initialization log lines; `--verbose` no longer sets it.
The full Java command is printed by `--show-command`, not by `--verbose`.
The summary is CLI-only; library callers can explicitly use `GenerationSession.prepare()`
to prepare a session and obtain its execution settings without advancing its position.

-----------

## 🔧 Embed in your own tools

`--show-command` prints the exact Java + JVM invocation used under the hood, so you can replicate it in IntelliJ, Maven, Gradle, or any launcher:

```bash
jitllm --gpu --model beehive-llama-3.2-1b-instruct-fp16.gguf \
  --prompt "tell me a joke" --show-command
```

Each command has focused help:

```bash
./jitllm --help
./jitllm run --help
./jitllm chat --help
./jitllm serve --help
./jitllm bench --help
```

| Command | Purpose | Key options |
| --- | --- | --- |
| `run` | Generate one response, then exit | `--prompt`, `--system-prompt`, `--max-new-tokens` |
| `chat` | Terminal conversation using one persistent session | `--system-prompt`, `--max-new-tokens` per turn |
| `serve` | OpenAI-compatible HTTP API | `--host`, `--port`; experimental `--continuous-batching` |
| `bench` | Repeated prefill/decode workloads | `--pp`, `--tg`, `--depth`, `--repetitions`, `--output` |

`./jitllm --help` lists every command and option; `./jitllm COMMAND --help` shows one command's.
Options are grouped as Engine Configuration (model, prompt, sampling, context, prefill mode,
`-v`/`--verbose`), the command's own group (Server, Benchmark), Hardware Configuration, Debug
and Profiling, TornadoVM Execution Verbose, and Advanced Options, where experimental options
are tagged `[experimental]`.

```bash
./jitllm run -m model.gguf --gpu -c 4096 --max-new-tokens 128 --prompt "Explain SIMD."
./jitllm chat -m model.gguf --gpu -c 4096 --max-new-tokens 128
./jitllm serve -m model.gguf --gpu -c 4096 --host 127.0.0.1 --port 8080 -v
./jitllm bench -m model.gguf --gpu --pp 128,512 --tg 64 --depth 0,4096 --repetitions 3 --output json
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


```bash
# Peek at what TornadoVM is doing
./jitllm --gpu --model model.gguf --prompt "..." --print-kernel      # generated GPU kernel
./jitllm --gpu --model model.gguf --prompt "..." --print-bytecodes   # TornadoVM bytecodes
./jitllm --gpu --model model.gguf --prompt "..." --debug --full-dump # everything
```
        
-----------

## 🙏 Acknowledgments

Partially funded by EU Horizon Europe & UKRI grants (most recent first):
[AERO 101092850](https://aero-project.eu/) · [P2CODE 101093069](https://p2code-project.eu/) · [ENCRYPT 101070670](https://encrypt-project.eu) · [TANGO 101070052](https://tango-project.eu).

## License

[MIT](LICENSE)
