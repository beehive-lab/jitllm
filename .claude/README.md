# JitLLM agents and skills

Performance procedures live in `.agents/skills/`. The matching `.claude/skills/` directories are relative symlinks to that source, so Claude Code and Codex use the same instructions. Keep canonical directories and links together when sharing the repository. Other existing build, model-porting and integration skills remain available.

Start with `jitllm-performance-analysis` for an end-to-end investigation. It loads only the relevant tools:

Its modes are **Analyze** (default: diagnose without production changes), **Compare** (measure/explain differences without fixes), and **Optimize** (diagnose, change, measure and verify when requested). Investigations maintain falsifiable hypotheses with proposed/supported/contradicted/unresolved statuses. Agents exchange compact evidence records linked to raw artifacts instead of rereading whole traces.

| Skill | Responsibility |
| --- | --- |
| `jitllm-benchmark` | Cold/startup, warmed forwards, end-to-end and paired measurement |
| `jitllm-java-analysis` | JFR, async-profiler, Java/JVM/native CPU, waits and memory |
| `jitllm-tornado-analysis` | Compilation, plans, transfers, residency and native interop |
| `jitllm-nsys-analysis` | CUDA streams/graphs, copies, waits and critical path |
| `jitllm-ncu-analysis` | Counters for identified consequential CUDA kernels |
| `jitllm-metal-analysis` | Apple tool discovery, Metal timelines/counters, native MLX |
| `jitllm-jmh` | Isolated measured Java costs |
| `jitllm-performance-verify` | Correctness, paired acceptance and experiment history |

Thin role definitions live in `.claude/agents/*.md` and `.codex/agents/*.toml`:

| Agent | Claude model | Responsibility |
| --- | --- | --- |
| `jitllm-benchmarking-specialist` | Sonnet | Controlled measurement |
| `jitllm-perf-profiling-specialist` | Inherit | Cross-layer diagnosis and tool selection |
| `jitllm-perf-optimizer` | Inherit | One evidence-backed candidate in assigned files |
| `jitllm-perf-verifier` | Sonnet | Independent acceptance/rejection/inconclusive verdict |

Codex roles inherit the session model; select a cheaper available Codex model at dispatch when supported. No Claude model alias is assigned to Codex. Verify the actual role/model in the client; if delegation is unavailable, run the roles sequentially. Measurements sharing an accelerator must be serialized; artifact analysis can be parallel.

Example request: “Use jitllm-performance-analysis to explain first-token latency and warmed decode on this model/backend. Separate Java, TornadoVM and device costs; optimize only after evidence and verify any candidate.” For measurement alone, invoke `jitllm-benchmark` directly.

Preserve raw evidence and failed experiments under ignored `perf-results/`; keep current machine settings and performance conclusions out of reusable instructions. No external plugin installation is required. Tool-specific captures require the corresponding installed profiler; Metal timelines need full Xcode with Instruments/`xctrace`. Without it, the Metal skill routes to task-level and host evidence and states the missing timeline coverage. JFRDoctor is optional when already connected: the local-tool-only Claude roles request its analysis from the parent rather than invoke MCP tools directly. Research/adoption decisions are in [sources.md](../.agents/skills/jitllm-performance-analysis/references/sources.md).

Deferred scenarios for behavioral evaluation are recorded in [evaluation-scenarios.md](../.agents/skills/jitllm-performance-analysis/references/evaluation-scenarios.md); recording them does not run or authorize an evaluation.

After changing skills/agents, reload or restart the clients and inspect their available skill/agent inventories. File-format checks do not establish live profiler availability or hardware validation.
