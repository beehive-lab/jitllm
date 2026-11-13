---
name: jitllm-performance-analysis
description: Investigate JitLLM slowdowns, unexplained performance regressions, host overhead and startup/prefill/decode phase costs when the cause or responsible layer is unknown. Use for requests such as why inference is slow, why decode got slower after a change, or how to investigate a bottleneck across Java, TornadoVM and CUDA/Metal. Coordinate focused profiling and distinguish measured differences from supported causes.
license: Apache-2.0
---

# End-to-end investigation

Define the target: cold launch/first token, load/initialization, compilation, prefill, decode at a given KV depth, serving latency/throughput, memory, or regression. Keep startup when it is the question. Choose tools from evidence; do not require the user to choose a profiler or supply a prior benchmark.

## Operating mode

Infer the mode from the request and record it before delegation. Default to **Analyze** when optimization is not requested; discovering a bottleneck does not authorize a fix. A change of mode requires user intent that authorizes its actions, not another agent's recommendation.

| Mode | Actions | Stop when |
| --- | --- | --- |
| Analyze | Capture, inspect and explain; no production-code changes. Necessary temporary diagnostic instrumentation belongs in an isolated copy and is excluded from acceptance measurements. | The requested performance question is answered with evidence, or remaining hypotheses are unresolved with precise missing evidence/next test. |
| Compare | Pin both commits/configurations/backends, measure comparable work and investigate differences; no production-code fixes. Disclose expected differences in hardware/path/timing scope. | Differences, uncertainty and supported explanations are reported; distinguish a measured delta from an explained cause. |
| Optimize | Diagnose, implement a scoped candidate, benchmark and verify. Require a supported hypothesis, change ownership and acceptance policy before implementation. | Verification accepts/rejects the candidate, or reports an inconclusive result and missing evidence. Further candidates need a distinct evidence-backed test. |

## Experiment contract

- Record exact commands/relevant JVM properties; JitLLM and TornadoVM commits and dirty patches; selected JAR/JDK/SDK/library versions; model identity/quantization; device/backend; actual execution path; native/graph selection; prompt tokens, output length, KV depth, concurrency and seed. Confirm correctness and no silent fallback. Discover flags in `jitllm` and `Options.java`, not old instructions.
- Preserve commands, logs, samples and raw traces in a unique ignored `perf-results/<experiment>/` or explicit scratch directory. Pass bounded summaries and artifact paths between agents. Locate symbols with `rg`; avoid ingesting source trees or whole traces.
- Record accelerator identity, other load, power/thermal state and cache/warmup policy. Serialize measurements sharing an accelerator, including Apple shared-memory pressure. Parallelize offline analysis. Reap only processes owned by this experiment.

## Load only the relevant skill

| Question | Skill |
| --- | --- |
| Baseline, A/B, timing scope | `jitllm-benchmark` |
| Java CPU, waits, allocations, GC, locks, I/O, native stacks | `jitllm-java-analysis` |
| Graph construction, device JIT, residency, dispatch, native interop | `jitllm-tornado-analysis` |
| CUDA streams, graphs, copies, synchronization, GPU gaps | `jitllm-nsys-analysis` |
| Named consequential CUDA kernel | `jitllm-ncu-analysis` |
| Apple Metal timeline/counters, native MLX | `jitllm-metal-analysis` |
| Isolated measured Java hotspot | `jitllm-jmh` |
| Optimization acceptance | `jitllm-performance-verify` |

Use a light baseline and focused host/device passes. Separate startup and in-process warmed execution. Distinguish HotSpot Java compilation, TornadoVM device compilation, driver/library compilation and first-use allocation. A prior CLI process warms some caches, not the next JVM's compiler state. Profiling can alter scheduling; keep a low-observer-overhead control and inspect whether metrics themselves enable profiling.

## Correlate and decide

Use explicit phase windows, PID/TID, task, stream/queue and dispatch IDs. Across tools/runs align markers and boundaries; do not assume shared clocks or identical sampled executions. State missing coverage. Kernel sums, CPU samples, API time and wall time have different denominators. Host overhead is not a subtraction: not wall time minus kernel, timer or GPU-busy totals, and never wall time from one run minus GPU time from another, because timing scope, overlap and execution conditions differ. Within one capture's phase window, GPU idle is exposed idle; attribute it to host work only with correlated submission, wait and CPU evidence.

Rank findings by exposed critical-path time and confidence. Each needs artifact/query and units, mechanism, alternative explanation, discriminating test and owner (Java/application, TornadoVM runtime/compiler, driver/library, kernel). GPU idle, long synchronization, low utilization/occupancy or one hot kernel alone cannot establish causality. Label Amdahl/roofline estimates as models, not measured gains.

## Hypothesis loop

Before collecting more data, state a falsifiable mechanism, its predicted observation and competing explanations. Choose the **smallest experiment that distinguishes them**, fixing workload/timing boundaries and changing one intended factor. State which possible results would support or contradict each explanation before running it. Read existing evidence first; do not repeat a capture unless it lacks the required coverage or the run contract changed.

| Hypothesis | Required evidence / discriminating experiment |
| --- | --- |
| Host dispatch limits decode | Correlate delayed submissions/GPU gaps with CPU stacks and scheduling in the same phase. |
| Device memory bandwidth limits decode | Measure representative kernel traffic/throughput and dependencies; compare controlled shapes/access patterns, not utilization alone. |
| TornadoVM recompiles identical operations | Compare first/reused executions with matching plan identity, shape/device and cache keys plus compiler events. |
| CUDA Graphs reduce launch overhead | Paired graph on/off with path liveness; compare CPU submission work, exposed GPU gaps and end-to-end time. |

After each experiment update each hypothesis: **proposed** (untested prediction), **supported** (evidence matches and distinguishes relevant alternatives), **contradicted** (valid evidence conflicts with its prediction), or **unresolved** (insufficient, conflicting or unavailable evidence). Confidence is separate from status. An unavailable counter or missing event cannot contradict a hypothesis. Revise/rank hypotheses, then select the next test only if it can change the decision. Stop according to the selected mode; do not gather more metrics merely because a tool is available.

## Delegation and evidence handoff

Assign `jitllm-benchmarking-specialist` bounded measurements, `jitllm-perf-profiling-specialist` diagnosis, `jitllm-perf-optimizer` candidate changes **only in Optimize mode**, and `jitllm-perf-verifier` independent acceptance. Each assignment carries mode, question/hypothesis IDs, run-contract path, inputs/artifacts, owned files/actions, target metric, stopping condition and any budget. The parent owns dispatch, shared-device access and resolution of contradictory findings; workers request follow-up roles through handoffs and cannot authorize scope expansion.

Use each named agent's configured model. Claude defaults are Sonnet for benchmarker/verifier and inheritance for profiler/optimizer; inheritance does not upgrade a weaker parent. For ambiguous causality/compiler/numerics, escalate to the parent with missing evidence and required expertise. In Codex use explicitly configured/available models or dispatch settings; do not translate Claude aliases or claim a model switch without observing it. If delegation is unavailable, keep the same role/mode boundaries sequentially. Never run profiling and acceptance measurements concurrently on one device.

Use this compact internal handoff in YAML or equivalent JSON; it is not a mandatory user-report layout. One record per hypothesis; preserve IDs/status transitions in the ignored experiment directory. Evidence entries require an artifact, a reproducible selector (query/event/thread/window) and an observation with units/denominator where numeric. Link command/tuple metadata once through `run_contract`; reread raw traces only to resolve missing or disputed evidence.

```yaml
finding:
  id: H1
  mode: analyze
  run_contract: perf-results/run-01/manifest.json
  phase: decode
  layer: java
  hypothesis: Host dispatch limits decode
  status: proposed
  prediction: Exposed GPU gaps coincide with delayed host submissions
  evidence: []
  confidence: low
  competing_explanations: [Device dependencies, Driver synchronization]
  next_experiment: Correlate submission cadence and CPU stacks in the decode window
  decision: pending
```

Benchmarker returns samples/validity/uncertainty in evidence; profiler updates hypothesis status and next test; optimizer adds candidate-diff/build paths and predicted effect; verifier adds correctness/performance artifacts and accepted/rejected/inconclusive decision. Preserve competing explanations, unsupported claims and coverage gaps across handoffs.

Return a phase/coverage table, ranked hypotheses with statuses/evidence and next action. In Optimize mode continue through verification; Analyze/Compare end without a fix. Read [references/sources.md](references/sources.md) only for upstream provenance/tradeoffs.

For skill-maintenance planning only, [references/evaluation-scenarios.md](references/evaluation-scenarios.md) records deferred behavioral tests; it does not authorize or initiate them.
