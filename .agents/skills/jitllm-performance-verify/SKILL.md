---
name: jitllm-performance-verify
description: Accept or reject JitLLM optimizations using correctness gates and paired low-overhead end-to-end measurements, preserving experiment history. Use after a candidate change or to audit an optimization claim.
license: Apache-2.0
---

# Evidence to acceptance

Require candidate diff/build, baseline identity, hypothesis, target metric/timing scope, workload and acceptance policy. Reconstruct missing context from artifacts or `jitllm-benchmark`. Use raw evidence independently of the optimizer's verdict.

Use the operating-mode and internal handoff contract in `jitllm-performance-analysis`. Auditing an existing optimization claim does not authorize new production changes. Preserve hypothesis IDs/statuses and add evidence selectors, tested build/diff paths, correctness/performance results and the acceptance decision; do not treat a supported diagnosis as proof that its proposed fix works.

1. Inspect the bounded diff and prove the intended path changes. Preserve unrelated edits. Acceptance excludes temporary instrumentation, debug/validation, altered GC/clock settings or artificial waits unless they are the actual proposed feature.
2. Select gates from `docs/architecture/verification.md`, `pom.xml` and affected tests. For kernel/precision/layout/transfer/async/native changes run relevant accelerator reference/parity and lifecycle tests on the target backend, including boundary shapes/depths. Fluent output is insufficient. Fix reference/tolerances before comparison; do not relax them for speed. Missing hardware/fixtures or skipped tests mean unverified correctness.
3. Use `jitllm-benchmark` for interleaved same-session A/B with identical work/timing scope. Inspect `scripts/perf-ab-run.sh` and `scripts/perf_gate.py`; the gate checks rates, not work. From each metrics file require all expected pairs, finite rates, matching actual token counts and per-side execution identity. Numeric-only inputs need separate work/path evidence. Add startup/prefill/memory/latency measures when targeted. Decode-only gates cannot prove them. Retain raw samples/spread/paired deltas; noisy/truncated/mismatched runs are inconclusive. A gate `PASS` establishes its regression floor only (the default allows a 3% slowdown); improvement requires the declared effect and uncertainty criteria separately.
4. Recheck the diagnosed mechanism in a separate focused diagnostic pass when needed. Kernel/JMH gains alone cannot prove end-to-end gain. Check phase/shape/backend regressions within the claimed scope; avoid unrelated matrices.
5. Record accepted, rejected or inconclusive with reasons. Stop when declared performance/correctness conditions pass and material regressions are resolved. Budget exhaustion or missing evidence is incomplete verification, not success.

## Experiment memory

Append `experiments.jsonl` in the ignored experiment directory for **every** candidate, including failures: ID/parent, hypothesis/technique, commit **and dirty patch/build identity**, JDK/SDK/library/device/workload tuple, timing boundary, command-file paths, raw sample/trace paths, correctness gates/results, decision policy and outcome. Preserve source snapshots/diffs to avoid retrying failed ideas. Read only matching history before the next candidate.

Reuse `docs/perf-history.jsonl`/gate tools for supported numeric tracking; do not overwrite history or append unverified profiler numbers. Historical tracking does not replace richer local records or paired measurement.

Return decision, effect/uncertainty, correctness/regression scope, artifacts, limitations and remaining work. Inconclusive is a valid outcome; it is not verified improvement.
