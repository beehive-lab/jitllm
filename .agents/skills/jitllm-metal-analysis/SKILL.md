---
name: jitllm-metal-analysis
description: Investigate JitLLM on Apple GPUs with Instruments/Metal timelines, CPU evidence and targeted counters, including native MLX calls. Gate tools on the actual Metal or OpenCL path.
license: Apache-2.0
---

# Apple host and GPU evidence

Use `jitllm-performance-analysis` for the contract. Detect SDK/backend/device/native provider: Apple hardware alone does not imply Metal. A Python MLX comparator is not a profile of Java/TornadoVM/native execution.

Record macOS/Xcode/architecture, power/thermal state and other load. Instruments timelines require a full Xcode installation with `xctrace`; Command Line Tools alone do not supply it. Probe `xcode-select -p`, `xcrun --find xctrace`, and, only when available, `xcrun xctrace list templates` / `xcrun xctrace help record`. Other optional tools require installed help establishing their capabilities. Do not switch/install Xcode globally as a profiling step.

If `xctrace` is unavailable, first use `jitllm-tornado-analysis` for the selected backend's supported task timers and their scope/coverage limits, then `jitllm-java-analysis` for host CPU/waits, then `--print-taskgraph-chain` to inspect plan structure. Run these as separate diagnostic passes. Task timers and a graph listing cannot recover queue overlap, exposed GPU gaps or hardware counters; report those as unobserved rather than infer a device bottleneck. Missing timeline tools need not block useful host or task-level analysis.

Select an available template with GPU/queue and host scheduling events (often Metal System Trace or Game Performance); Time Profiler answers host-only questions. Launch capture must cover child Java, not just Python. For warmed execution attach to the correct Java PID with a bounded window; late attach cannot recover startup.

Adapt to installed syntax/template:

```bash
xcrun xctrace record --template "$TEMPLATE" --output "$OUT/apple.trace" \
  --time-limit 30s --attach "$JAVA_PID"
xcrun xctrace export --input "$OUT/apple.trace" --toc --output "$OUT/toc.xml"
```

Inspect TOC before exporting selected tables/XPath; resolve XML ID references and retain units/time origins. Do not invent schemas. A timeline `.trace` differs from debugger `.gputrace`. If counters require interactive capture or are unavailable, preserve useful timeline evidence and state the limit.

- Correlate Java dispatch, TornadoVM compilation/pipeline creation, native MLX, command buffers/encoders, queues/resources, waits and GPU work using IDs/phase markers. Inspect `backend/tornado/ops/nativelib/apple` and the actual SDK library provider when implicated.
- Empty Metal lanes on OpenCL mean missing visibility, not zero GPU time. Use supported OpenCL task profiling plus Java/OS evidence; do not promise Metal counter coverage.
- Use interval unions within explicit phase windows of one capture for busy/idle/overlap. Idle is exposed GPU idle, not host overhead, until correlated with host evidence; never subtract GPU busy time from another run's wall time. Low utilization may mean small work, dispatch/dependency gaps, memory limits or wrong capture scope; it does not prove work is outside MLX.
- Unified memory still has coherence, faults/migration, residency and CPU/GPU contention. Correlate resource events/VM/RSS; do not add overlapping host/device memory views.
- Collect counters for consequential commands only. Counter sets vary by hardware/toolchain; no fixed SIMD/occupancy/bandwidth threshold proves a bottleneck. Counter replay can serialize passes. Acceptance excludes debug/validation layers and heavy counters.
- For MLX comparators explicitly evaluate lazy work and synchronize at timing boundaries with that runtime's supported APIs. For Java native MLX verify provider completion/async semantics; do not insert Python `mx.eval` advice or a wait per operation. Compare identical host/device timing scope.

Return commands/artifacts, phase/queue dependencies, counter coverage/limits, mechanism and next test. Resolve host gaps with `jitllm-java-analysis`; accept changes through `jitllm-performance-verify`.

References: [Metal tools](https://developer.apple.com/metal/tools/), [performance analysis](https://developer.apple.com/documentation/xcode/analyzing-the-performance-of-your-metal-app/), [counter serialization](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics).
