# Native image and AOT cache

Two ways to start jitllm faster than a plain JVM: a GraalVM native image (no JVM at run time) or a
Leyden AOT cache (JDK 25+, keeps HotSpot). Both need a **jdk22plus** TornadoVM SDK in `TORNADOVM_HOME`.
GPU kernels are compiled by TornadoVM at run time in both cases, and cached in
`$TORNADOVM_HOME/var/cuda-codecache` — keep `TORNADOVM_HOME` exported when running, or the cache goes to
`./null/var` in the current directory.

## Native image

```bash
export GRAALVM_HOME=<GraalVM for JDK 25 with native-image>
./mvnw -Pnative -DskipTests package          # target/jitllm, plus the shared libraries next to it
                                             # (libjsvml.so on Linux, libsleef.dylib on macOS): ship them together
```

`scripts/native-image.sh build` does the same without Maven. Run the image with the system
properties the `jitllm` launcher would pass, plus the library path:

```bash
./target/jitllm -Duse.tornadovm=true -Djava.library.path=$TORNADOVM_HOME/lib \
    -Dtornado.tvm.maxbytecodesize=65536 -Xmx20g -m model.gguf -p "Tell me a joke" --instruct
```

The metadata ships in the jar, so any native image with jitllm on its class path (a Quarkus native
application, for one) picks it up:

| File | What it covers |
|---|---|
| `META-INF/native-image/io.github.beehive-lab/jitllm/native-image.properties` | builder flags: FFM with shared arenas, `-H:-VectorAPISupport` (incompatible with shared arenas), `-Djitllm.VectorBitSize=0` with `FloatTensor` initialized at build time (the CPU path runs scalar code), and `JitllmNativeImageFeature` |
| `JitllmNativeImageFeature` | scans jitllm's classes at build time: every class loadable by name, every class with task lambdas (`$deserializeLambda$`) registered as lambda-capturing — TornadoVM resolves kernels through `SerializedLambda` |
| `…/jitllm/reachability-metadata.json` | jitllm's class files as resources: TornadoVM compiles kernels from their bytecode |
| `…/jitllm/tornadovm/reachability-metadata.json` | what TornadoVM 8 and the JDK need (reflection, resources, FFM downcalls), recorded on the CUDA, OpenCL and Metal backends. TornadoVM ships no metadata yet; refresh it with `scripts/native-image.sh trace` after a TornadoVM upgrade |

Every TornadoVM backend calls its native library through FFM downcalls, and an image can only make the
downcalls it was built with. A backend missing from the metadata fails on its first call
(`MissingForeignRegistrationError`), which jitllm reports as "no accelerator device is available".
`trace` merges what one machine's runs record into the file, so run it once on each backend: on a CUDA
machine, an OpenCL one and a Mac with Metal.

On Metal the image runs, but decodes several times slower than the JVM on the same machine (Qwen3-0.6B
F16, Apple Silicon: 25 tok/s against 149); CUDA images decode at least as fast as the JVM. The cost is
per token, in TornadoVM's Metal driver on the host side. F16 models are ready to generate as soon as
on the JVM; Q8_0 models a little later, since the image decodes their weights for MLX more slowly
(Llama-3.2-3B Q8_0: 5.4 s against 3.9 s).

The image is built without the Vector API: GraalVM rejects `-H:+VectorAPISupport` together with shared
arena support ("Support for Arena.ofShared is not available with Vector API support"), and TornadoVM
opens shared arenas (`CUDAContext`, `OCLContext`, `CuBlasLtNativeLib`). Only jitllm's CPU path uses the
Vector API; the GPU path is unaffected. In the image the CPU path runs scalar code instead: the
properties file sets `-Djitllm.VectorBitSize=0` and initializes `FloatTensor` at build time, so the
width is fixed when the image is built and `-Djitllm.VectorBitSize` at run time has no effect. CPU
inference in an image is 6-23× slower than on the JVM (Qwen3-0.6B and Llama-3.2-1B, on an RTX 5070 Ti
machine and on Apple Silicon). This applies to any native image built with the jitllm jar, a Quarkus
application included; use the GPU path, or the JVM for CPU inference.

Profile-guided optimization (Oracle GraalVM) builds only with sampling: `--pgo-instrument` fails in
the shared-arena phase (`After inserting all session checks call Unsafe.convEndian was not inlined and
could access a session`, at a different `MemorySegment` access in each build, TornadoVM's included).
`--pgo-sampling` works: add it to the `native-image` command, run the image on each model you serve
with `-XX:ProfilesDumpFile=<model>.iprof`, then rebuild with `--pgo=<the .iprof files, comma-separated>`.
On the 19 models below it gains nothing over the plain Oracle image (startup +3%, prefill +2%, decode
+1.5%, all within run-to-run noise), so it is not part of the default build.

## AOT cache (JDK 25+)

```bash
scripts/aot-cache.sh train target/qwen3.aot -m Qwen3-0.6B-f16.gguf -p "Tell me a joke" --instruct
scripts/aot-cache.sh run   target/qwen3.aot -m Qwen3-0.6B-f16.gguf -p "Write a haiku" --instruct
```

Train with the workload you run; training and production need the same JDK, jar and JVM options
(`JVM_OPTS` adds options to both). On JDK 25 the cache holds loaded and linked classes and method
profiles but no compiled code, so TornadoVM's kernel compilation still starts cold.

The cache does not store the resolved module graph for jitllm (the JVM logs "archivedBootLayer not
available, disabling full module graph"). JDK 25 archives the boot layer only when it contains no
incubator module, and jitllm's contains `jdk.incubator.vector` twice over: jitllm adds it for its CPU
Vector API code, and the TornadoVM SDK keeps `tornado-unittests.jar` in `share/java/tornado`, which
provides a `TornadoLibraryProvider` service that `tornado.runtime` uses, so service binding pulls
`tornado.unittests` (which requires `jdk.incubator.vector`) into the boot layer. Without both, the
graph is stored and saves about 50 ms per start (the same module set on a trivial main: 20 ms against
68 ms); against jitllm's 2-3 s startup that is about 2%.

## Measured (RTX 5080 Laptop, 2026-11-03)

TornadoVM develop 733af6562 (CUDA), JDK 25.0.2, Oracle GraalVM 25.0.5, GraalVM CE 25.2.4; 19 GGUF
models (Qwen2.5, Qwen3, Llama-3.2, granite-4.0, gemma-4, Phi-3, Mistral-7B; Q4_0, Q8_0 and F16), a
530-token prompt, 128 greedy tokens, `--ctx-size 1024`, warm kernel and page caches, median of 3
launches. Startup is launch until the `-v` run configuration is printed; TTFT is launch until the first
generated byte. The JVM runs with `-Xms20g -Xmx20g`, the images with `-Xmx20g`. Every configuration
generates the same text.

Geometric mean over the 19 models, relative to the JVM:

| | startup | TTFT | prefill | decode | peak RSS | faster startup |
|---|---|---|---|---|---|---|
| JVM + AOT cache (classes only) | −7% | −6% | 1.00× | 1.00× | −0% | 18 of 19 |
| JVM + AOT cache (full) | −6% | −5% | 0.99× | 0.99× | −0% | 17 of 19 |
| native image, Oracle GraalVM | −41% | −40% | 1.59× | 1.11× | −15% | 19 of 19 |
| native image, GraalVM CE | −30% | −30% | 1.52× | 1.10× | −14% | 19 of 19 |

Startup, seconds:

| model | JVM | JVM + AOT cache (classes only) | native (Oracle) | native (CE) |
|---|---|---|---|---|
| Qwen3-0.6B F16 | 2.00 | 1.79 | 1.03 | 1.22 |
| Llama-3.2-1B F16 | 1.93 | 1.83 | 1.12 | 1.31 |
| gemma-4-E2B Q8_0 | 2.84 | 2.72 | 1.76 | 2.44 |
| Phi-3-mini Q8_0 | 2.42 | 2.27 | 1.32 | 1.55 |
| Qwen3-4B Q8_0 | 2.63 | 2.44 | 1.60 | 1.91 |
| Mistral-7B Q8_0 | 2.61 | 2.51 | 1.61 | 1.75 |

The full AOT cache adds nothing over the classes-only cache on JDK 25: startup is dominated by
TornadoVM's kernel JIT and device setup, which the cache does not cover. The images' higher prefill and
decode throughput is mostly the JVM's host code still warming up during a 128-token run; it shrinks in a
long-running process.

Avoid element-by-element loops over large TornadoVM arrays on the startup path (`init`, `set`/`get` in a
copy loop). Identical Oracle GraalVM builds sometimes run each `MemorySegment` access through a method
handle instead of a compiled store; before jitllm stopped re-zeroing its KV cache, such builds took about
22 ms per MiB of cache (Phi-3: 11 s to ready instead of 2.4 s). Use the arrays as allocated (they are
zero) and `MemorySegment.copy` for bulk copies.
