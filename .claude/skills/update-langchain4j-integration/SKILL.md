---
name: update-langchain4j-integration
description: Update the LangChain4j jitllm integration to a new jitllm release. Use when bumping io.github.beehive-lab:jitllm, adapting the adapter to API or capability changes, enabling inherited integration tests, validating JDK/TornadoVM combinations and demos, or updating the module documentation.
---

# Update the LangChain4j jitllm integration

Treat a version bump as an integration migration. Discover current repository state and preserve
unrelated changes.

## Inputs

Determine or ask for:

- jitllm base version
- jitllm, LangChain4j, and demo checkout paths
- exact GGUF paths for chat and capability-specific tests
- supported JDK/TornadoVM backend combinations

Never guess paths, versions, models, flags, or backends.

## 1. Inspect the release

Read the jitllm changelog, release diff, POM, and changed public APIs. Confirm both artifacts:

- `jitllm:<version>-jdk21`
- `jitllm:<version>-jdk22plus`

For a published release:

```bash
/path/to/jitllm/.claude/skills/update-langchain4j-integration/scripts/inspect-release.sh <version>
```

Otherwise build and inspect both JARs locally. Among others check model loading/generation, metrics,
`ChatFormat`, tool/thinking support, stop tokens, and TornadoVM lifecycle APIs.

## 2. Inspect and update LangChain4j

The integration is the `langchain4j-jitllm` module (package `dev.langchain4j.model.jitllm`,
classes `JitLLMChatModel`, `JitLLMStreamingChatModel`, `JitLLMBaseModel`, `JitLLMConversions`,
`JitLLMResponseParser`), added upstream in langchain4j/langchain4j#6508. The older
`langchain4j-gpu-llama3` module targets GPULlama3.java and is not updated for jitllm releases.
Re-read the layout rather than assuming it:

```bash
rg -n "jitllm|jdk21|jdk22plus" pom.xml langchain4j-bom/pom.xml langchain4j-jitllm/pom.xml
```

Read the module POM (its `jdk21`/`jdk22plus` profiles select the artifact and `jitllm.version`),
the adapter classes, the inherited IT subclasses, `src/test/resources/tornado-jvm.args`, the module
README, the docs page `docs/docs/integrations/language-models/jitllm.md` and the BOM entry.

Keep changes minimal and backend agnostic:

- use `TORNADOVM_HOME`; leave backend configuration to the selected SDK. `onGPU(true)` must not
  name a backend: `ModelOptions.backend(CUDA)` makes the engine reject an OpenCL or Metal SDK, so the
  adapter leaves it unset and relies on `-Duse.tornadovm=true`
- the JDK profiles select `-jdk21`/`-jdk22plus`; never hardcode either
- use `MODEL` for tests; never hardcode a GGUF path
- preserve inherited ITs and enable only implemented capabilities
- keep unsupported feature combinations disabled with accurate reasons

Three adapter rules this integration has already had to learn:

- **A `ChatModel` is stateless.** The caller owns the conversation and sends all of it on
  every request, while an engine session retains its own history. Reset the session per
  request, or each one appends the whole conversation to the previous and the context grows
  without bound.
- **Import only `api/**`.** `BackendId` and `ExecutionPolicy` are reachable and marked
  experimental; anything else from the engine is an internal type and a rule violation. Keep
  `@Experimental` types (`GenerationTimings`, for one) out of the module's public signatures.
- **A truncated response is still a response.** Return the generated text with
  `FinishReason.LENGTH` on `MAX_TOKENS`/`CONTEXT_FULL`, never a substitute notice
  (langchain4j#5959 fixed this in the GPULlama3 module).

For a new capability, implement both request and response mappings, including conversation
history, metadata, finish reasons, and streaming callbacks. Add deterministic unit tests where
possible, then enable inherited ITs incrementally.

Tool calling does not imply forced/named tool choice or structured-output support. Verify calls
with arguments, without arguments, multiple calls, tool results followed by final answers, and
sync/streaming callbacks separately.

## 3. Validate

Use focused tests while iterating, preferably in fresh TornadoVM processes when diagnosing shared
device state. Classify failures as adapter, model behavior, jitllm, TornadoVM, or test harness.

Completion requires the entire sequence in
[whole-chain-validation.md](references/whole-chain-validation.md) for every claimed JDK/backend:

1. set up Java and TornadoVM
2. build the required LangChain4j modules
3. run the full module ITs
4. build the demos
5. run the required demos

Compilation or focused tests alone do not complete the update.

## 4. Curate and report

Update only affected documentation with commands that actually passed. Document supported
versions, matching SDKs, preview requirements, `TORNADOVM_HOME`, `MODEL`, and unsupported
capabilities. Remove stale versions, machine paths, generated classpaths, and obsolete commands.

Finish with:

```bash
git diff --check
./mvnw -pl langchain4j-jitllm spotless:check
git status --short
```

Report the API changes, exact JDK/backend/models tested, every whole-chain result, failures,
disabled capabilities, publication compatibility, and files that must not be committed. Remember
that source profiles do not make one flattened, published LangChain4j POM dual-JDK compatible.
Never check a PR checklist item for tests that were not run.
