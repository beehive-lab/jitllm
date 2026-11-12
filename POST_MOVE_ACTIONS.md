# Post-move actions — GPULlama3.java → jitllm

What is left of the rename, all of it **outside this repository** or in binary assets: each item
needs a person with access to another system. Tick the boxes as they land, then delete the file.

> **Naming recap.** `io.github.beehive-lab` (groupId) is unchanged. `gpu-llama3` → `jitllm`
> (artifactId). `org.beehive.gpullama3` → `org.beehive.jitllm` (Java package and module).
> `beehive-lab/GPULlama3.java` → `beehive-lab/jitllm` (repo). `llama-tornado` → `jitllm`,
> `llamaTornado` → `jitllm4j` (launchers). `~/.gpullama3` → `~/.jitllm`. Releases are tagged
> `jitllm-vX.Y.Z`. This is a **clean break**: no relocation POM, no aliases.

---

## Open items

| ✓ | Item | Action |
|---|---|---|
| [ ] | **LangChain4j** docs page | The README links `docs.langchain4j.dev/integrations/language-models/gpullama3-java`. Re-point it once the upstream page for the `langchain4j-jitllm` module exists. |
| [ ] | **Quarkus LangChain4j** docs page | The README links `docs.quarkiverse.io/.../gpullama3-chat-model.html`. Re-point it once upstream publishes the JitLLM page. Upstream config keys `quarkus.langchain4j.gpu-llama3.*` are public API: let them sequence any rename with a deprecation cycle. |
| [ ] | `beehive-lab/docker-gpullama3.java` | It clones the old repo and runs `./llama-tornado`. Update the clone URL and the launcher to `./jitllm`, decide whether new tags go to a `beehivelab/jitllm` repository, then add a Docker badge back to the README. |
| [ ] | Hugging Face collections (`*-gpullama3java`) | Slugs stay — renaming 404s the old ones. Update descriptions and repo links only. |
| [ ] | CLA-assistant | `CONTRIBUTING.md` links `cla-assistant.io/beehive-lab/jitllm`; confirm the CLA is registered against `beehive-lab/jitllm`. |
| [ ] | `docs/ll.gif` (README hero) | A terminal recording showing `./llama-tornado`. Re-record with `./jitllm`, or drop the embed. |
| [ ] | `docs/performance.png` | "GPULlama3.java" is rendered into the pixels and nothing references it. Re-render or delete. |
| [ ] | Citation | Consider minting a Zenodo DOI at the next release, so existing citations of `jitllm (formerly GPULlama3.java)` stay linked. |

---

## Do not do

- **Never create a repo at an old slug** (`beehive-lab/GPULlama3.java`, `beehive-lab/jllm`).
  GitHub's redirect survives until someone claims that name, at which point every old link,
  clone URL and issue reference breaks permanently.
- **Never rewrite `docs/perf-history.jsonl`.** It is an append-only
  historical record; old workflow names in it are evidence, not stale
  references.
- **Never rename the `beehive-lab` org.** It also owns TornadoVM and backs the verified
  Maven Central namespace `io.github.beehive-lab`.
