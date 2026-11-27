# Whole-chain validation

The verification source of truth. Run all five stages for every JDK/backend you claim.

## Inputs

```bash
export LANGCHAIN4J_DIR=/path/to/langchain4j
export DEMO_DIR=/path/to/gpullama3-langchain4j-demo   # a copy switched to langchain4j-jitllm
export MODEL=/exact/path/to/model.gguf
```

Never commit a model path.

## 1. Java and TornadoVM

```bash
export JAVA_HOME=/path/to/jdk-21          # or a JDK 22+ (CI uses 25)
export TORNADOVM_HOME=/path/to/tornadovm-<version>-<backend>
export PATH="$JAVA_HOME/bin:$TORNADOVM_HOME/bin:$PATH"

java -version
"$TORNADOVM_HOME/bin/tornado" --version
"$TORNADOVM_HOME/bin/tornado" --devices
test -f "$MODEL"
```

The Java and TornadoVM lines must match: an SDK built with `make BACKEND=...` is the JDK 21
line, and one built with `make jdk22plus BACKEND=...` serves JDK 22 and up, which is what a
JDK 25 run needs. Export in the same shell command as whatever uses them — each tool call is
a fresh shell. `sdk use` is interactive-only; set `JAVA_HOME`/`PATH` explicitly instead.

## 2. Build the modules

```bash
cd "$LANGCHAIN4J_DIR"
./mvnw \
  -pl langchain4j-bom,langchain4j-agentic,langchain4j-jitllm \
  -am clean install \
  -DskipTests -DskipITs -Drevapi.skip=true

./mvnw -pl langchain4j-jitllm dependency:tree \
  -Dincludes=io.github.beehive-lab:jitllm
```

Confirm the resolved artifact ends in the suffix the active JDK implies — `-jdk21` on JDK 21,
`-jdk22plus` on JDK 22+. The module's own `jdk21`/`jdk22plus` profiles do that; there is no flag.

`attach-javadocs` runs with doclint on and warnings fatal. Every public and protected member
needs a comment, and a method with a return value needs `@return`. A missing one fails
`install`, not just the javadoc jar.

## 3. Run the suite

```bash
/path/to/jitllm/.claude/skills/update-langchain4j-integration/scripts/validate-langchain4j-integration.sh \
  "$LANGCHAIN4J_DIR"
```

Three environment facts the suite depends on, all of which fail in ways that look like
adapter defects:

- `src/test/resources/tornado-jvm.args` must carry `--add-modules jdk.incubator.vector`.
  Without it every test dies in `FloatTensor`'s static initializer with
  `NoClassDefFoundError: jdk/incubator/vector/Vector`, before any assertion runs. The SDK's
  own flags do not supply it.
- The device budget in the same file must fit the whole suite, not one model. It stands up
  several models and sessions in one JVM, and device memory a closed session frees goes back
  to TornadoVM's buffer provider rather than to the driver — so the later classes fail on
  allocation while the earlier ones passed. 20GB is enough on a 24GB device.
- TornadoVM SDKs after 7.0.1 ship JUnit 6 in `share/java/tornado`, and `tornado.runtime`
  binds `tornado.unittests`, so every TornadoVM JVM resolves the JUnit Platform and the
  Jupiter engine as named modules. The suite's classpath copies of `junit-jupiter-params`,
  and the `junit-vintage` engine bundled in `junit-platform-console-standalone`, cannot read
  their internals. The run then fails with `TestEngine with ID 'junit-vintage' failed to
  discover tests`, or every parameterized container fails with `IllegalAccessError: ... does
  not export org.junit.platform.commons.util` while the plain tests pass. The fix
  (langchain4j#7295) is `--add-modules org.junit.jupiter.params` in `tornado-jvm.args` and
  `--exclude-engine=junit-vintage` in the `run-tests` launcher arguments. Check what the SDK
  resolves with `tornado --jvm="--show-module-resolution" -version | grep junit`.

The run-tests JVM sets `-Duse.tornadovm=true`, so a unit test that asserts `onGPU(true)` fails
without TornadoVM must be disabled there (`@DisabledIfSystemProperty`), not counted as a failure.

To run one inherited test, select its class and filter by method name; `--select-method`
does not find a method declared in LangChain4j's abstract base class:

```bash
../mvnw -P run-tests "-Dit.selection=--select-class=dev.langchain4j.model.jitllm.JitLLMChatModelIT \
  --include-methodname=.*should_execute_a_tool_then_answer.*"
```

### The inherited tests measure the model as much as the adapter

Several of LangChain4j's shared contract tests assert model behaviour. Which of them pass
depends on the fixture, and no single small fixture passes all of them.

The ITs build their models with `.think(false)`, and jitllm rejects `ThinkingMode.DISABLED`
for a model without a reasoning phase (`thinkingMode(DISABLED) was requested, but LLAMA_3 has
no reasoning phase to control`). Until the ITs set it only for thinking models, use a
fixture with a reasoning phase, such as Qwen3; a Llama fixture fails every generating test
on that error, not on its output.

Measured with unit tests included (the script's selection); re-measure on the current
release before relying on it:

| jitllm / TornadoVM | Fixture | Result |
| --- | --- | --- |
| 3.1.0 / 8.2.0 | `Qwen3-0.6B-F16` | 75/77 on JDK 25 Metal and JDK 21 Metal (Apple M4 Pro); 3 skipped |
| 1.0.0 / 7.0.1 | `Qwen3-0.6B-F16` | 42/42 on JDK 25 CUDA, JDK 21 CUDA and JDK 25 OpenCL |
| 1.0.0 / 7.0.1 | `Llama-3.2-3B-Instruct-Q8_0` | 36/42: the six tool tests call the tool correctly, but the final answer lacks the expected text |

The two 3.1.0 Metal failures are `should_execute_a_tool_then_answer`, sync and streaming: the
model answers the weather question itself instead of calling the tool. It does the same on
the CPU and on jitllm 1.0.2, so it is fixture behaviour, not a regression.

Before recording a failure as an adapter defect, establish which it is: rerun the test on the
CPU (`-Duse.tornadovm=false` in a copy of `tornado-jvm.args`) and on the previous release in a
separate worktree. Identical output on both points to the model. Tool transport is covered
deterministically by `JitLLMConversionsTest`, whatever a given fixture emits.

**Report the fixture with the counts, and never call the inherited suite green.**

## 4. Build the demos

```bash
cd "$DEMO_DIR"
mvn clean package dependency:build-classpath -Dmdep.outputFile=cp.txt -DskipTests
mvn dependency:tree \
  -Dincludes=dev.langchain4j:langchain4j-jitllm,io.github.beehive-lab:jitllm
```

Confirm the local artifacts resolved rather than a stale release. Do not commit `cp.txt`.

## 5. Run the demos

Read the current commands and classes first — do not reuse remembered ones:

```bash
cd "$DEMO_DIR"
rg -n "tornado|--params|public static void main" README.md src/main/java
```

Run basic chat, streaming chat, the capability this release changed, and any long-running
demo through to its end. Startup is not a pass: each demo must perform inference and reach
its expected end.

The demo repository still targets `langchain4j-gpu-llama3`. Validate on a copy: set
`langchain4j.version` to the LangChain4j snapshot you built, switch the dependency to
`langchain4j-jitllm`, the imports to `dev.langchain4j.model.jitllm.JitLLM*`,
`agentBuilder(...).outputName(...)` to `outputKey(...)`, and pass `MODEL` as the model path.
Delete the `printLastMetrics()` calls in `_1_ChatExample` and `_2_StreamingExample`: that
method belongs to the GPULlama3 adapter, and the JitLLM adapter has never had it.

Launch with `tornado --jvm="-Duse.tornadovm=true --add-modules jdk.incubator.vector ..."`, and
add `--enable-preview` on JDK 21. `TicTacToeAgenticGame` takes `--gpu-model` and `--cpu-model`
and runs one model on each, so it covers the CPU path too.

Small fixtures run the demos to their end, but they can still give poor answers; for
example, `Qwen3-0.6B-F16` mixes up the user's name in `_3_ChatMemoryExample`. Record that,
but it is not a failure.

## Record

Java and TornadoVM versions, backend and device, resolved dependencies, the fixture, test
counts with the fixture that produced them, build results, and every demo command with its
result.
