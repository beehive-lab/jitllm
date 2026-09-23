# Whole-chain validation

The verification source of truth. Run all five stages for every JDK/backend you claim.

## Inputs

```bash
export LANGCHAIN4J_DIR=/path/to/langchain4j
export DEMO_DIR=/path/to/gpullama3-langchain4j-demo
export MODEL=/exact/path/to/model.gguf
```

Never commit a model path.

## 1. Java and TornadoVM

```bash
export JAVA_HOME=/path/to/jdk-21          # or a JDK 25
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
  -pl langchain4j-bom,langchain4j-agentic,langchain4j-gpu-llama3 \
  -am clean install \
  -DskipTests -DskipITs -Drevapi.skip=true

./mvnw -pl langchain4j-gpu-llama3 dependency:tree \
  -Dincludes=io.github.beehive-lab:jitllm
```

Confirm the resolved artifact ends in the suffix the active JDK implies — `-jdk21` on JDK 21,
`-jdk25` on JDK 25. The module's own `jdk21`/`jdk25` profiles do that; there is no flag.

`attach-javadocs` runs with doclint on and warnings fatal. Every public and protected member
needs a comment, and a method with a return value needs `@return`. A missing one fails
`install`, not just the javadoc jar.

## 3. Run the suite

```bash
/path/to/jitllm/.claude/skills/update-langchain4j-integration/scripts/validate-langchain4j-integration.sh \
  "$LANGCHAIN4J_DIR"
```

Two environment facts the suite depends on, both of which fail in ways that look like
adapter defects:

- `src/test/resources/tornado-jvm.args` must carry `--add-modules jdk.incubator.vector`.
  Without it every test dies in `FloatTensor`'s static initializer with
  `NoClassDefFoundError: jdk/incubator/vector/Vector`, before any assertion runs. The SDK's
  own flags do not supply it.
- The device budget in the same file must fit the whole suite, not one model. It stands up
  several models and sessions in one JVM, and device memory a closed session frees goes back
  to TornadoVM's buffer provider rather than to the driver — so the later classes fail on
  allocation while the earlier ones passed. 20GB is enough on a 24GB device.

### The inherited tests measure the model as much as the adapter

Several of LangChain4j's shared contract tests assert model behaviour. Which of them pass
depends on the fixture, and no single small fixture passes all of them:

| Fixture | Result |
| --- | --- |
| `Qwen3-0.6B-f16` | 15/19 — the tool tests pass; `should_respect_multiple_messages` fails |
| `Llama-3.2-3B-Instruct-Q8_0` | 11/19 — `should_respect_multiple_messages` passes; the six tool tests fail |

Before recording either as an adapter defect, establish which it is. The multi-message
failure is not one: the engine receives and encodes the whole conversation, which the 0.6B
model's own reasoning trace confirms by repeating the earlier turn before declining to
answer. Tool transport is covered deterministically by `GPULlama3ConversionsTest`, whatever
a given fixture emits.

**Report the fixture with the counts, and never call the inherited suite green.**

## 4. Build the demos

```bash
cd "$DEMO_DIR"
mvn clean package dependency:build-classpath -Dmdep.outputFile=cp.txt -DskipTests
mvn dependency:tree \
  -Dincludes=dev.langchain4j:langchain4j-gpu-llama3,io.github.beehive-lab:jitllm
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

## Record

Java and TornadoVM versions, backend and device, resolved dependencies, the fixture, test
counts with the fixture that produced them, build results, and every demo command with its
result.
