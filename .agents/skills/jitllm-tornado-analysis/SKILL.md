---
name: jitllm-tornado-analysis
description: Analyze TornadoVM profiler dumps/JSON and task timers in JitLLM, including graph construction, compilation, dispatch, transfers, residency and native-library interop. Use when interpreting TOTAL_KERNEL_TIME or TASK_KERNEL_TIME, reconciling native-library host timers with GPU device timers, or asking whether TornadoVM profiler measurements explain host overhead or prefill/decode phase costs.
license: Apache-2.0
---

# TornadoVM boundary

Verify JDK/SDK/backend/device and actual path. Use installed SDK profiler/event definitions; timing coverage and semantics vary, especially for native tasks.

## Focused code entry points

Java paths below are relative to `src/main/java/org/beehive/jitllm/`. Locate relevant symbols, not entire directories.

| Concern | Read first |
| --- | --- |
| Load/compile/generate | `model/loader/ModelLoader.java`, `inference/TokenGenerationLoop.java`, `runtime/metrics/MetricKey.java` |
| Plan selection/graph/setup | `backend/tornado/TornadoVMMasterPlan*.java`, `backend/tornado/plan` |
| Reuse/dispatch/completion | `backend/tornado/lowering/{CompiledProgramCache,SharedWorkspacePlan,InvocationBoundary}.java` |
| Metrics producer | `backend/tornado/TornadoMetricsReporter.java`, `auxiliary/RunMetrics.java` |
| Residency/workspace/KV | `backend/tornado/{memory,workspace,kv,tensor}`, `backend/tornado/plan/layout` |
| Native dispatch/fallback | `backend/tornado/NativeLibrarySupport.java`, `backend/tornado/ops/nativelib/{nvidia,apple}` |
| Multi-GPU stage/stream handoff | `backend/tornado/pipeline/PipelineTransport.java` and selected transport |

Discover TornadoVM source checkout separately from `TORNADOVM_HOME` (installed SDK). Follow the selected compiler/runtime/library-provider symbols, not every backend. Record dependency builds/diffs with the experiment.

## Capture and reconcile

Discover current flags in `jitllm`; a focused pass starts with `--profiler --profiler-dump-file "$OUT/tornado.json" --verbose` on the requested workload. Summarize:

```bash
python3 scripts/summarize_tornado_profiler.py "$OUT/tornado.json" --top 20
```

The helper sums reported timers by graph-name phase hints; it is not a host/device critical-path timeline. Names do not establish phase: it labels shared graphs such as `layer_*`, `logits` and `activationUpdate` as decode even when they ran in sequential prefill or a warm-up pass, and graph families it does not list as `other`. Establish each execution's phase from execution order and the plan source. Its phase percentages divide graph totals by the task-timer sum; do not quote them as shares. Inspect raw fields for compiler, dispatch, copies and libraries. Verify units, cumulative/per-execution semantics, missing fields/counts and actual phase names. Do not double-count graph totals and task totals. Missing/zero may mean unsupported instrumentation.

Check the profiler implementation at the source revision matching the running SDK. In the inspected `ed4731325` checkout, `CUDAInstalledCode` waits for each generated kernel's event when profiling outside capture; `TornadoVMInterpreter` times native-library calls with host `nanoTime` between synchronized stream markers. Those task timers mix device-event durations and host wall durations and must not be ranked as homogeneous device time. Timing is disabled during graph capture, and `executeGraphLaunch` does not populate ordinary per-task timers, so replay can be missing from the breakdown. These are revision-scoped facts, not a promise about every SDK; verify them before interpreting a new version. Use profiler output to locate reported work, and reconcile each timer's scope with timeline evidence rather than substituting it for unprofiled wall time. Host overhead is not a subtraction: not wall time minus task-timer totals, and never wall time from one run minus GPU busy time from another, because timing scope, overlap and execution conditions differ. Attribute it from correlated host and device activity within one low-overhead capture: graph-level on CUDA (`jitllm-nsys-analysis`) or `jitllm-metal-analysis` on Apple. Gaps between graph launches are exposed GPU idle until matched to host submissions, waits and CPU samples. Rank kernels with `--cuda-graph-trace=node` only in a separate diagnostic capture.

Separate graph construction, TornadoVM device compilation, driver/module/pipeline compilation, allocation/first-touch, conversion/upload and execution. TornadoVM 8.0+ uses TC (`tornado-compiler/`), replacing the older Graal/JVMCI device compiler; legacy profiler key names alone do not identify the compiler. CLI setup/JIT buckets can include host setup; upload buckets can include prepacking. Inspect boundaries before labeling them pure compiler or DMA time. Compare first use to reuse in the same process. New shapes/devices/plans can legitimately compile; prove identical cache keys/plan identity repeat before alleging a caching defect.

For per-step transfers identify buffer, bytes/direction/frequency, ownership/lifetime, transfer mode, synchronization and consumer. Include native prepacking/dequantization, representation conversion, workspace and readback. Test residency changes for stale state and memory cost. Unified/zero-copy can still fault, migrate or wait; device memory metrics may exclude mapped weights.

Check `SharedWorkspacePlan.invoke` locking/logits copying for serving serialization; distinguish lock contention from device waits. Check actual native-provider queue/completion behavior for async calls. Do not add waits everywhere and call the altered schedule representative.

Profiler mode and even an enabled metrics sink can activate event timing (`TornadoMetricsReporter.enableOn`), forcing waits/changing overlap. Compare a low-overhead control; correlate `jitllm-java-analysis` with `jitllm-nsys-analysis` or `jitllm-metal-analysis`. Kernel totals cannot explain unobserved wall time. Return coverage, responsible symbols, mechanism, competing hypothesis and next test/change owner.
