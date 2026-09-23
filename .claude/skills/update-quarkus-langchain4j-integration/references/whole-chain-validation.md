# Whole-chain validation

This is the verification source of truth. Run all five stages for every claimed JDK/backend.

## Inputs

```bash
export QUARKUS_LANGCHAIN4J_DIR=/path/to/quarkus-langchain4j
export DEMO_DIR=/path/to/Quarkus-Langchain4j-GPULlama3-Demos
export MODEL=/exact/path/to/model.gguf
```

Use an exact tool-capable model path for tool demo runs when needed. Never commit model paths.

## 1. Set up Java and TornadoVM

```bash
sdk use java 25.0.2-open
sdk use tornadovm 4.0.0-jdk25-ptx

java -version
test -n "$TORNADOVM_HOME"
"$TORNADOVM_HOME/bin/tornado" --version
"$TORNADOVM_HOME/bin/tornado" --devices
test -f "$MODEL"
```

These are the versions recorded as working for this integration; re-check them against the
current README/CLAUDE.md before reusing, since supported versions can change with a jitllm
bump. `sdk use` is interactive-only — if driving from a non-interactive shell, export
`JAVA_HOME`/`PATH` explicitly instead of relying on `sdk use`.

The Java and TornadoVM JDKs must match. Obtain any required native-library environment from the
selected SDK; do not put backend-specific paths in project configuration.

## 2. Build the required quarkus-langchain4j modules

```bash
cd "$QUARKUS_LANGCHAIN4J_DIR"
./mvnw install \
  -pl model-providers/gpu-llama3/runtime,model-providers/gpu-llama3/deployment \
  -am -DskipTests

./mvnw -pl model-providers/gpu-llama3/runtime dependency:tree \
  -Dincludes=io.github.beehive-lab:jitllm
```

Confirm the resolved jitllm artifact matches the active JDK (`-jdk21` vs `-jdk25`, selected by
the root `jdk25` Maven profile based on the JDK running the build).

## 3. Run the module tests

```bash
/path/to/jitllm/.claude/skills/update-quarkus-langchain4j-integration/scripts/validate-quarkus-langchain4j-integration.sh \
  "$QUARKUS_LANGCHAIN4J_DIR"
```

This stage passes only when tests are discovered, at least one runs, Surefire reports zero
failures/errors, and the command exits successfully. Use isolated tests only for diagnosis, then
rerun this stage.

## 4. Build the demos

```bash
cd "$DEMO_DIR"
mvn clean install
```

Confirm the demos resolve the local `quarkus-langchain4j-gpu-llama3` and matching `gpu-llama3`
artifacts installed in step 2, not stale versions from an earlier local repository cache. The
demos repository pins both `gpu-llama3.version` and `quarkus.version` in its root POM, and both
drift: a stale `gpu-llama3` pin surfaces as `ClassNotFoundException` on a facade type that the
extension compiled against, and a Quarkus version that disagrees with the extension's surfaces
as `TypeNotPresentException: io/quarkus/arc/impl/TypeVariableImpl` during bean generation.
Neither is an extension defect; check the pins before investigating anything else.

## 5. Run the required demos

Read the current README and demo sources first — commands and modules change over releases:

```bash
cd "$DEMO_DIR"
cat README.md
find demos -maxdepth 1 -type d
```

Run the documented TornadoVM commands for at least:

1. `chat-demo` (basic chat)
2. `streaming-demo` (streaming chat)
3. the demo exercising the capability changed by this release (e.g. `tool-demo-ls`,
   `java-coder-demo`), when applicable

Do not reuse old jar names, model parameters, or JVM options without verifying them against the
current README. Startup alone is not a pass; each demo must perform inference and reach its
expected end — a completed response, not just server boot.

The extension calls `System.setProperty("tornado.device.memory", ...)` from its own
`device-memory` config during model initialization, so a bare `-Dtornado.device.memory` on the
command line is overwritten. Raise
`-Dquarkus.langchain4j.gpu-llama3.chat-model.device-memory` instead.

## Record

Record Java/TornadoVM versions, backend/device, resolved dependencies, model, test counts, build
results, and every demo command/result.
