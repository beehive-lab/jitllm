# Whole-chain validation

The verification source of truth. Run every stage for every JDK/backend you claim.

```bash
export SPRING_AI_DIR=/path/to/spring-ai
export MODEL=/exact/path/to/model.gguf     # never commit a model path
MODULES=models/spring-ai-jitllm,auto-configurations/models/spring-ai-autoconfigure-model-jitllm,starters/spring-ai-starter-model-jitllm
```

Export `JAVA_HOME`/`PATH` in the same shell command as whatever uses them; `sdk use` is
interactive-only.

## 1. Build and checks, per JDK

```bash
cd "$SPRING_AI_DIR"
export JAVA_HOME=/path/to/jdk-25; export PATH="$JAVA_HOME/bin:$PATH"   # then again with a JDK 21
./mvnw -B -pl spring-ai-bom,spring-ai-model install -DskipTests -Dmaven.javadoc.skip=true
./mvnw -B -pl "$MODULES" -Dmaven.build.cache.enabled=false clean install
./mvnw -B -q -pl models/spring-ai-jitllm dependency:tree -Dincludes=io.github.beehive-lab:jitllm
javap -v -cp models/spring-ai-jitllm/target/classes org.springframework.ai.jitllm.JitLlmChatModel | grep major
```

Expect 0 Checkstyle violations, NullAway clean, javadoc clean, `-jdk22plus` on JDK 22+ and
`-jdk21` on JDK 21, and class major 65 on both. Disable the build cache
(`-Dmaven.build.cache.enabled=false`) or checkstyle and tests may be reported "cached".

`clean install` runs the unit tests and, with `MODEL` set, the real-model ITs on the CPU.

## 2. Real-model ITs on the GPU

```bash
/path/to/jitllm/.claude/skills/update-spring-ai-integration/scripts/validate-spring-ai-integration.sh \
  "$SPRING_AI_DIR" /path/to/jdk /path/to/tornadovm-sdk "$MODEL"
```

It runs every `*Tests` and `*IT` of the model and autoconfigure modules in a TornadoVM JVM
(`-Duse.tornadovm=true`, plus `--enable-preview` on JDK 21) and fails unless both report zero
failures. Build the modules on the same JDK first. Sample `nvidia-smi` during the run to confirm
the GPU is used.

Measured on jitllm 1.0.0 / TornadoVM 7.0.1, RTX 4090:

| JDK / SDK | Model | spring-ai-jitllm | autoconfigure |
| --- | --- | --- | --- |
| 25 / jdk22plus-cuda | Qwen3-0.6B-F16 | 37/37 | 5/5 |
| 21 / jdk21-cuda | Qwen3-0.6B-F16 | 37/37 | 5/5 |
| 25 / jdk22plus-opencl | Qwen3-0.6B-F16 | 37/37 | 5/5 |
| 25 / jdk22plus-cuda | Llama-3.2-3B-Instruct-Q8_0 | 36/37 | 4/5 |

The Llama failures are the tool round trip: the tool is called correctly, but the 3B model's final
answer ignores its result. That is model behaviour, not the adapter. Report the model with the
counts.

## 3. Spring Boot smoke app from the starter

```bash
/path/to/jitllm/.claude/skills/update-spring-ai-integration/scripts/run-boot-smoke.sh \
  /path/to/jdk /path/to/tornadovm-sdk "$MODEL"
```

It builds `scripts/boot-smoke` (only `spring-ai-starter-model-jitllm` plus the Spring AI BOM,
installed locally by stage 1) and runs it through `tornado` on the GPU. Each line must appear:
`[call]` with an answer, `[meta]` with `on-gpu=true` and a token rate, `[stream]` with the streamed
answer, `[tool] getWeather(...)` and `[tools]` using the tool's result. On JDK 21 the script
swaps in `jitllm:<version>-jdk21` the way `jitllm-chat.adoc` tells users to.

## Record

Java and TornadoVM versions, backend and device, resolved jitllm artifact, the model, test counts
with the model that produced them, and the smoke-app output.
