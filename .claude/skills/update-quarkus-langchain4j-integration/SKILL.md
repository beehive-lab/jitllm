---
name: update-quarkus-langchain4j-integration
description: Update the Quarkus LangChain4j jitllm extension to a new jitllm release. Use when bumping io.github.beehive-lab:jitllm in quarkus-langchain4j, adapting the runtime/deployment modules to API or capability changes, validating JDK/TornadoVM combinations and demos, or updating module documentation. See also the sibling skill update-langchain4j-integration, which targets the upstream dev.langchain4j integration instead of this Quarkus extension.
---

# Update the Quarkus LangChain4j jitllm integration

Treat a version bump as an integration migration. Discover current repository state and preserve
unrelated changes.

## Inputs

Determine or ask for:

- jitllm base version
- jitllm, quarkus-langchain4j, and demo checkout paths
- exact GGUF paths for chat and capability-specific tests
- supported JDK/TornadoVM backend combinations

Never guess paths, versions, models, flags, or backends.

## 1. Inspect the release

Read the jitllm changelog, release diff, POM, and changed public APIs. Confirm both artifacts:

- `jitllm:<version>-jdk21`
- `jitllm:<version>-jdk22plus`

For a published release:

```bash
/path/to/jitllm/.claude/skills/update-quarkus-langchain4j-integration/scripts/inspect-release.sh <version>
```

Otherwise build and inspect both JARs locally. Among others check model loading/generation, metrics,
`ChatFormat`, tool/thinking support, stop tokens, and TornadoVM lifecycle APIs.

## 2. Inspect and update quarkus-langchain4j

Locate the integration rather than assuming its current layout:

```bash
rg -n "gpu-llama3" pom.xml model-providers/gpu-llama3
```

The module lives at `model-providers/gpu-llama3/{runtime,deployment}`, and the version comes from
the `gpu-llama3.version` property in the root `pom.xml`: the default picks `-jdk21`, and the
`jdk25` profile overrides it to `-jdk25`. The branch this project validates against is
`jitllm/facade-1.0.0` on the maintainer's fork, which is what jitllm's own CI clones (see
QUARKUS_LANGCHAIN4J_REF in .github/workflows/build-and-run.yml).
Re-read the current root `pom.xml` before editing it — do not assume this layout is unchanged.

Read the module POMs, `GPULlama3BaseModel`/`GPULlama3ChatModel`/`GPULlama3StreamingChatModel`,
`GPULlama3ResponseParser`, config classes, existing unit tests under
`model-providers/gpu-llama3/runtime/src/test`, the module README, and the root JDK profile.

Keep changes minimal and backend agnostic:

- use `TORNADOVM_HOME`; leave backend configuration to the selected SDK
- keep the `-jdk21`/`-jdk22plus` split driven by the root `jdk22plus` Maven profile, not hardcoded
- never hardcode a GGUF path in tests; use an environment variable
- preserve existing unit tests and add new ones for changed capabilities
- keep unsupported feature combinations disabled with accurate reasons

Three rules this extension has already had to learn:

- **Name no backend.** The engine rejects an explicit `BackendId` that disagrees with the
  device it resolves, rather than silently running on another one, so naming CUDA stops the
  extension starting on an OpenCL or Metal SDK. Set `use.tornadovm` and let the SDK decide.
- **Batched prefill is default-off.** It is default-off in the engine on every backend, and
  a default that turns it on makes every deployment opt in without asking. `prefill-decode`
  defaults to `false` and `prefill-batch-size` to `1`.
- **Import only `api/**`.** `BackendId` and `ExecutionPolicy` are reachable and marked
  experimental; anything else from the engine is an internal type and a rule violation.

For a new capability, implement both request and response mappings, including conversation
history, metadata, finish reasons, and streaming callbacks. Add deterministic unit tests where
possible.

Tool calling does not imply forced/named tool choice or structured-output support. Verify calls
with arguments, without arguments, multiple calls, tool results followed by final answers, and
sync/streaming callbacks separately.

## 3. Validate

Use focused tests while iterating, preferably in fresh TornadoVM processes when diagnosing shared
device state. Classify failures as adapter, model behavior, jitllm, TornadoVM, or test harness.

Completion requires the entire sequence in
[whole-chain-validation.md](references/whole-chain-validation.md) for every claimed JDK/backend:

1. set up Java and TornadoVM
2. build the required quarkus-langchain4j modules
3. run the module tests
4. build the demos
5. run the required demos

Compilation or focused tests alone do not complete the update.

## 4. Curate and report

Update only affected documentation with commands that actually passed. Document supported
versions, matching SDKs, preview requirements, `TORNADOVM_HOME`, the model path variable, and
unsupported capabilities. Remove stale versions, machine paths, generated classpaths, and obsolete
commands.

Any `@ConfigItem`/config-interface change (new property, changed `@WithDefault`, changed Javadoc,
`Optional` vs primitive) drifts `docs/modules/ROOT/pages/includes/quarkus-all-config.adoc` and the
module-specific `quarkus-langchain4j-gpu-llama3*.adoc` files, which are generated, not hand-edited.
CI's `Documentation check` job runs `mvn clean install` and fails the build if
`git status --porcelain docs/modules/ROOT/pages/includes/` is non-empty afterwards. Regenerate and
commit these before opening/updating the PR:

```bash
cd "$QUARKUS_LANGCHAIN4J_DIR"
./mvnw clean install \
  -pl model-providers/gpu-llama3/runtime,model-providers/gpu-llama3/deployment,docs \
  -am -DskipTests -DskipITs -Drevapi.skip=true

git status --short docs/modules/ROOT/pages/includes/
git diff docs/modules/ROOT/pages/includes/
```

Review the diff: it should only reflect the config change just made.

`quarkus-all-config.adoc` aggregates every module, so a `-pl ... -am` build regenerates it
from the subset that was built and deletes the sections for everything else. That is a
partial-build artifact, not drift. Take the module-specific `quarkus-langchain4j-gpu-llama3*.adoc`
files from the regeneration, revert `quarkus-all-config.adoc`, and apply the same edits to its
gpu-llama3 rows by hand — or run a full `mvn clean install`, which is what CI does. Do not commit
the deletions.

Any genuinely unrelated hunk in the module files means the docs were already out of sync before
this change: regenerate anyway and call it out in the report rather than hand-editing them.

Finish with:

```bash
git diff --check
git status --short
```

Report the API changes, exact JDK/backend/models tested, every whole-chain result, failures,
disabled capabilities, and files that must not be committed. Never check a PR checklist item for
tests that were not run.
