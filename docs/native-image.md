# Native image and AOT cache

Two ways to start jitllm faster than a plain JVM: a GraalVM native image (no JVM at run time) or a
Leyden AOT cache (JDK 25+, keeps HotSpot). Both need a **jdk22plus** TornadoVM SDK in `TORNADOVM_HOME`.
GPU kernels are compiled by TornadoVM at run time in both cases, and cached in
`$TORNADOVM_HOME/var/cuda-codecache` — keep `TORNADOVM_HOME` exported when running, or the cache goes to
`./null/var` in the current directory.

## Native image

```bash
export GRAALVM_HOME=<GraalVM for JDK 25 with native-image>
./mvnw -Pnative -DskipTests package          # target/jitllm (+ target/libjsvml.so, ship both)
```

`scripts/native-image.sh build` does the same without Maven. Run the image with the system
properties the `jitllm` launcher would pass, plus the library path:

```bash
./target/jitllm -Duse.tornadovm=true -Djava.library.path=$TORNADOVM_HOME/lib -Xmx20g \
    -m model.gguf -p "Tell me a joke" --instruct
```

The metadata ships in the jar, so any native image with jitllm on its class path (a Quarkus native
application, for one) picks it up:

| File | What it covers |
|---|---|
| `META-INF/native-image/io.github.beehive-lab/jitllm/native-image.properties` | builder flags: FFM with shared arenas, `-H:-VectorAPISupport` (incompatible with shared arenas; the CPU path's Vector API runs its portable Java code), and `JitllmNativeImageFeature` |
| `JitllmNativeImageFeature` | scans jitllm's classes at build time: every class loadable by name, every class with task lambdas (`$deserializeLambda$`) registered as lambda-capturing — TornadoVM resolves kernels through `SerializedLambda` |
| `…/jitllm/reachability-metadata.json` | jitllm's class files as resources: TornadoVM compiles kernels from their bytecode |
| `…/jitllm/tornadovm/reachability-metadata.json` | what TornadoVM 8 and the JDK need (reflection, resources, FFM downcalls). TornadoVM ships no metadata yet; regenerate with `scripts/native-image.sh trace` after a TornadoVM upgrade |

## AOT cache (JDK 25+)

```bash
scripts/aot-cache.sh train target/qwen3.aot -m Qwen3-0.6B-f16.gguf -p "Tell me a joke" --instruct
scripts/aot-cache.sh run   target/qwen3.aot -m Qwen3-0.6B-f16.gguf -p "Write a haiku" --instruct
```

Train with the workload you run; training and production need the same JDK, jar and JVM options
(`JVM_OPTS` adds options to both). On JDK 25 the cache holds loaded and linked classes and method
profiles but no compiled code, so TornadoVM's kernel compilation still starts cold. The JDK also
disables the cache's full module graph for jitllm ("archivedBootLayer not available"); the cause is
still open.

## Measured (RTX 5080 Laptop, Qwen3-0.6B F16, 128 tokens, warm kernel cache, 2026-11-02)

| | wall | ready to generate | decode tok/s |
|---|---|---|---|
| JVM (JDK 25.0.2) | 4.43 s | 3.52 s | 265 |
| JVM + AOT cache | 4.17 s | 3.23 s | 262 |
| native image | 3.42 s | 2.70 s | 264 |
