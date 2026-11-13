---
name: jitllm-java-analysis
description: Profile JitLLM host execution with JFR, async-profiler and JVM diagnostics to distinguish CPU, waits, HotSpot JIT, GC, allocations, locks, I/O and native calls from device bottlenecks.
license: Apache-2.0
---

# Java and native host evidence

Use the contract in `jitllm-performance-analysis`. Find the Java child PID/JDK actually executing `jitllm`; the launcher may be Python. Capture startup with JVM startup options; attaching later cannot recover initialization/JIT. Inspect launcher option propagation.

| Symptom | Smallest useful pass |
| --- | --- |
| Startup, compiler, GC, safepoints, threads, I/O | JFR `profile`; inspect enabled events/thresholds/loss |
| Java/JVM/native execution hotspot | async-profiler CPU with Java+native stacks |
| Latency, blocked dispatch, idle GPU | wall + CPU evidence + GPU timeline |
| Allocation/GC pressure | JFR allocation samples/GC or separate allocation pass |
| Contention/parking | monitor/park events or lock pass; distinguish device waits |
| Native/RSS growth | heap/NMT plus RSS/maps and driver/device evidence |

Discover versions/events/engine/attach constraints with `asprof --help`, `jfr help`, `jcmd "$JAVA_PID" help`. OS/architecture/JDK affect capabilities: Linux counters are not portable to macOS, and native/kernel stack visibility varies. Unsupported events mean missing coverage. Avoid machine security changes as routine workarounds.

Example startup JFR, with an existing writable `OUT` without spaces, `BACKEND_FLAG` set to the selected `--cuda`/`--metal`/`--opencl`, and the requested workload:

```bash
JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:+$JAVA_TOOL_OPTIONS }-XX:StartFlightRecording=filename=$OUT/host.jfr,settings=profile,dumponexit=true" \
  ./jitllm --gpu "$BACKEND_FLAG" --model "$MODEL" --prompt "$PROMPT" \
    --ctx-size "$CONTEXT" --max-new-tokens "$TOKENS" --seed "$SEED"
jfr summary "$OUT/host.jfr"
```

For live capture use `jcmd`'s `JFR.start`/`JFR.dump`/`JFR.stop`, discovering their arguments via help. Export relevant event types/windows, not whole recordings. A bounded CPU pass:

```bash
asprof -e cpu -d 30 -f "$OUT/cpu.html" "$JAVA_PID"
```

Choose event/duration for the phase; collect wall/alloc/lock separately when necessary. For short startup use the installed async-profiler's documented startup agent. Preserve TornadoVM launch options. Frame-pointer/JIT-symbol changes may affect codegen: disclose diagnostic settings, then verify without them.

## Attribution rules

- CPU samples locate execution; wall samples include blocked/sleeping threads and may sum across threads beyond elapsed time. Compare thread states/stacks. JNI/FFM/driver calls can wait for GPU completion; a long native call does not prove native CPU work. Correlate CUDA/Metal activity.
- Separate HotSpot compiler threads/class loading/GC from TornadoVM device compilation and driver compilation. JFR compiler events do not automatically cover all three. Locate relevant symbols in `api`, `inference`, `runtime`, `backend/tornado`, `auxiliary`; use `jitllm-tornado-analysis` for the boundary.
- Sample counts depend on interval, utilization and window. Few execution samples during GPU waiting can be correct; no fixed samples/second threshold proves bad capture. Missing symbols/disabled events/sub-threshold I/O limit conclusions. Exceptions need cost evidence, not just counts.
- Allocation samples are not retained heap. Heap, RSS, direct buffers, NMT and GPU/unified residency are different views. NMT needs startup enablement and excludes third-party native allocations; it cannot account for all TornadoVM/MLX/CUDA memory. Avoid double-counting Apple shared allocations.
- If already connected, JFRDoctor's nine analyzers can return compact JSON. A Claude role with a local-only tool allowlist requests that analysis through a handoff to the parent; reading this skill does not grant MCP access. Inspect only relevant categories with `framework=other`; verify coverage/units/thresholds against raw events. Its native/wait classification uses method-name heuristics, not measured CPU. JDK tools are the fallback; this skill requires no MCP installation.

Return phase/thread evidence, event counts/units, symbol quality, limits and ranked hypotheses with a GPU correlation or next test. Use `jitllm-jmh` only for an isolated operation, not end-to-end GPU proof.

References: [async-profiler engines](https://github.com/async-profiler/async-profiler/blob/master/docs/CpuSamplingEngines.md), [JFR](https://docs.oracle.com/en/java/javase/25/docs/specs/man/jfr.html), [NMT scope](https://docs.oracle.com/en/java/javase/25/vm/native-memory-tracking.html).
