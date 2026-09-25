---
name: update-spring-ai-integration
description: Create or update the Spring AI jitllm integration (spring-ai-jitllm, spring-ai-autoconfigure-model-jitllm, spring-ai-starter-model-jitllm) for a jitllm release. Use when bumping io.github.beehive-lab:jitllm in spring-ai, adapting JitLlmChatModel to API or capability changes, validating JDK/TornadoVM combinations on a GPU, running the Spring Boot smoke app, or updating the Spring AI docs page. See also update-langchain4j-integration and update-quarkus-langchain4j-integration.
---

# Update the Spring AI jitllm integration

Treat a version bump as an integration migration. Discover current repository state and preserve
unrelated changes.

## Inputs

Determine or ask for:

- jitllm base version
- jitllm and spring-ai checkout paths
- exact GGUF paths (a small reasoning model such as `Qwen3-0.6B-F16` and a non-reasoning one such
  as `Llama-3.2-3B-Instruct-Q8_0`)
- the TornadoVM SDKs to validate: one per JDK line (`jdk21`, `jdk22plus`) and per backend

Never guess paths, versions, models, flags, or backends.

## 1. Inspect the release

Read the jitllm changelog, release diff and `org.beehive.jitllm.api`. Confirm both artifacts with
the LangChain4j skill's script (it checks `-jdk21` is Java 21 and `-jdk22plus` Java 22 bytecode):

```bash
/path/to/jitllm/.claude/skills/update-langchain4j-integration/scripts/inspect-release.sh <version>
```

Check what changed in `GenerationRequest` (sampling, stop sequences, tools, events),
`GenerationResult` (finish reasons, tool calls, timings), `ChatMessage`/`ChatContent`/`ChatRole`,
`ModelOptions` and session lifecycle.

## 2. Inspect and update Spring AI

The integration was proposed in spring-projects/spring-ai as a new set of modules (it
supersedes the GPULlama3 proposal, #6840). Re-read the layout rather than assuming it:

```bash
rg -n "jitllm" pom.xml spring-ai-bom/pom.xml spring-ai-model/src/main/java/org/springframework/ai/model/SpringAIModels.java \
  spring-ai-docs/src/main/antora/modules/ROOT/nav.adoc
ls models/spring-ai-jitllm auto-configurations/models/spring-ai-autoconfigure-model-jitllm \
  starters/spring-ai-starter-model-jitllm
```

- The version is `jitllm.base.version` in the root `pom.xml`. The modules add the JDK suffix:
  `-jdk22plus` by default, `-jdk21` through the `jitllm-jdk21` profile. Bump only the base.
- The modules build only under the root `jitllm` profile (JDK 21+); Spring AI's baseline and its
  release/snapshot jobs are JDK 17, so they are not built or published there. Do not change that
  without the maintainers.

Rules this integration has already had to learn:

- **Import only `org.beehive.jitllm.api`** (plus `runtime.backend.BackendId`, experimental). Keep
  `@Experimental` types out of the public Spring API.
- **Never name a GPU backend.** `onGpu(true)` leaves `ModelOptions.backend` unset and requires
  `-Duse.tornadovm=true`; naming CUDA makes the engine reject an OpenCL or Metal SDK.
- **A `ChatModel` is stateless.** One session for the model's life, `reset()` before every
  request, the whole conversation every time. A session per request exhausts device memory.
- **Tool execution is not the model's.** Return `AssistantMessage.ToolCall`s with
  `TOOL_CALLS`; `ToolCallingAdvisor` executes them. Streaming with tools answers in one chunk.
- **A truncated response is still a response:** full text, `LENGTH`, a warning.
- **Publication:** Spring AI flattens POMs (`ossrh`), which drops profiles, so the published POM
  carries `-jdk22plus` whatever JDK built it. Compile to Java 21 bytecode (`java.version=21`) so
  the jar runs on both lines; JDK 21 users swap in `-jdk21` (documented in `jitllm-chat.adoc`).
  The autoconfigure module declares jitllm itself (optional) so JDK 21 builds pin `-jdk21`.
- Every autoconfigure dependency must be `<optional>` (enforcer), and the starter needs
  `spring-boot-starter`.

## 3. Validate

Completion requires every stage of [whole-chain-validation.md](references/whole-chain-validation.md)
for every claimed JDK/backend: build and checks, unit tests on the CPU, the real-model ITs on the
GPU through `tornado`, and the Spring Boot smoke app from the starter. Compilation or unit tests
alone do not complete the update. Classify failures as adapter, model behaviour, jitllm,
TornadoVM or harness.

## 4. Curate and report

Update `jitllm-chat.adoc` (and `comparison.adoc` only if a capability changed) with properties
and commands that actually passed. Finish with:

```bash
./mvnw -pl models/spring-ai-jitllm,auto-configurations/models/spring-ai-autoconfigure-model-jitllm,starters/spring-ai-starter-model-jitllm \
  spring-javaformat:apply
git diff --check
git status --short
```

Spring AI requires a DCO `Signed-off-by` on every commit, matching the commit author, and a human
accountable for AI-assisted changes. Sign off (`git commit -s`, with the maintainer's own git
identity) only when the maintainer has asked you to sign as them; otherwise commit without it and
leave the sign-off and the PR to them.
Report the API changes, exact JDK/backend/models and the counts each produced, the smoke-app
output, failures, unsupported capabilities, and publication caveats.
