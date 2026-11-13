---
name: jitllm-benchmark
description: Measure JitLLM throughput/latency baselines, paired A/B differences and backend comparisons with explicit cold-start, warmed forward-pass or end-to-end timing scope. Use for requests to measure or compare performance; requests to explain an unexplained slowdown or regression belong to jitllm-performance-analysis.
license: Apache-2.0
---

# Reproducible measurement

Use the contract in `jitllm-performance-analysis`. Measure the requested tuple; avoid unsolicited model matrices/context lengths/flag sweeps. Prove JAR selection, backend/path and flag liveness. Rebuild only when needed using `build-n-run-engine` / `build-tornado`.

Respect the assigned mode and hypothesis test. Return samples, run validity, uncertainty and reproducible evidence selectors using the orchestrator's compact handoff. A measured difference alone does not establish its cause or authorize a production fix.

## Choose the boundary

| Target | Inspect before running |
| --- | --- |
| Whole launch/first token | `jitllm`; Java `inference/TokenGenerationLoop.java`, `auxiliary/RunMetrics.java`, `auxiliary/metrics/{RunMetricsSnapshot,JsonMetricsRenderer}.java` |
| In-process warmed forwards, pp/tg/KV depth/synthetic | Java `bench/JitllmBench.java`; `./jitllm --bench --help` |
| Paired commits or JVM properties | `scripts/perf-ab-run.sh`, `scripts/perf_gate.py` paired mode |
| Historical tracking | `scripts/perf-gate-run.sh`, `scripts/perf-gate-tolerances.json` |
| Requested full backend matrix | `scripts/benchmark_backends.sh` (rebuilds dependencies/selects JDK) |

Java paths are relative to `src/main/java/org/beehive/jitllm/`. Forward-only benchmarks exclude tokenization/sampling/host argmax; CLI decode timers include more host work. Serving latency needs the serving entrypoint and specified arrivals/concurrency; forward tok/s cannot substitute.

CLI harnesses start new JVMs per repetition. Discarded CLI runs warm OS/driver caches, not the following JVM's HotSpot/in-memory plan cache. `--cache-cold` labels a run; it does not evict caches. Establish steady-state convergence **within** the measured process. Record cold JVM/compiler/driver/filesystem policies separately.

The gate launchers pass `MAX_TOKENS` as `--max-tokens`, the launcher's legacy spelling of `--ctx-size`: it sets context capacity (prompt plus generated tokens), not a generated-token limit, so EOS or the context limit decides each run's generated count. Inspect the launchers and `scripts/perf_gate.py` before relying on them. The gate compares `eval_rate` values, skips metrics files without a numeric rate, and in paired mode notes unequal run counts but still returns a verdict; it does not check token counts, timing scope or execution identity. Before comparing, read every metrics file: all expected repetitions/pairs present, finite positive `eval_rate`, matching `prompt_eval_count`/`eval_count`/`has_prefill_phase` across pairs, and recorded `execution_path`/`execution_combination`. `--samples` inputs carry no work or path evidence. Execution paths may intentionally differ between A/B sides, but must be identified and stable within each side. Metrics without these fields are incomplete evidence, not a pass.

Inspect `auxiliary/metrics/RunMetricsSink.java` and the renderer plus harness environment handling. Direct runs should preserve existing JVM options; scripts may replace `JAVA_TOOL_OPTIONS`, requiring `EXTRA_JVM_PROPS` or per-side properties. Do not edit the SDK argument file globally. An enabled device metrics sink may activate TornadoVM profiling: inspect `TornadoMetricsReporter.enableOn` and quantify observer overhead rather than call all metrics runs uninstrumented. JSON output alone does not establish that this sink is enabled.

## Compare fairly

- Use the existing gate procedure for comparable tuples; otherwise predeclare metric/repetitions/decision policy. Retain individual samples, central statistic and spread; increase repetitions when uncertainty matters. Interleave paired baseline/candidate runs to control thermal/order drift.
- Match prompt tokens, output length, KV depth, recurrent-state reset and timing scope. Record EOS/context truncation; unequal work invalidates comparisons. Pair seeds and validate outputs. A flag comparison changes one intended factor; a cumulative ladder is useful only for stacked optimizations.
- Require asynchronous work to complete at the timing boundary. Do not measure enqueue alone or force synchronization after every kernel in the control. Acceptance runs exclude heavyweight profilers, debug/validation layers and diagnostic JVM/codegen changes.
- Do not kill unrelated Java processes or delete unrelated JARs for idle/fresh state. The launcher may choose by filename rather than mtime: verify the JAR that actually runs.

Return build/commands, scope, actual work, raw samples/units, spread, paired deltas/policy, correctness and artifact paths. The decode gate validates its metric only; add startup/prefill/tail-latency/memory measurements when relevant. Use an analysis skill when causality is needed.

A gate `PASS` means the rate stayed above its regression floor; the default tolerance permits a 3% slowdown. It does not establish improvement. Apply the predeclared improvement/effect and uncertainty criteria separately before claiming an optimization worked.
