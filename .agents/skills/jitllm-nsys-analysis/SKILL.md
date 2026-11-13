---
name: jitllm-nsys-analysis
description: Capture or analyze a JitLLM CUDA timeline to attribute host dispatch, compilation, native libraries, copies, streams, graph replay, synchronization and GPU gaps. Use before kernel counters.
license: Apache-2.0
---

# CUDA timeline

Use `jitllm-performance-analysis` for the experiment contract and `jitllm-java-analysis` for Java stacks. CUDA tracing does not observe OpenCL/Metal device execution; CPU/OS views may still help.

1. Verify CUDA execution and child Java PID. Record `nsys --version`, `nsys profile --help`, `nsys stats --help-reports`. Trace capabilities/library APIs vary. Capture the launcher process tree and confirm Java CUDA events before interpreting empty traces.
2. Include startup for load/JIT/first-token questions. For steady state use in-process warmup and explicit phase markers/timestamps. Prior CLI runs do not warm this JVM; do not subtract initialization from another run.
3. Begin with a short pass; add supported cuBLAS/cuDNN tracing for native attribution. Verify installed JVM/JIT-symbol, CPU-sampling and context-switch options. A low-overhead CUDA-only pass can precede a focused host pass.

Example after defining output and requested workload variables:

CUDA graphs are enabled by default on the CUDA path. Nsight Systems defaults to graph granularity when available: this shows whole-graph activity and gaps, without individual node activities. Keep that low-overhead view for graph timing; do not rank decode kernels from its kernel summary.

```bash
nsys profile --trace=cuda,nvtx,osrt --output "$OUT/timeline" \
  ./jitllm --gpu --cuda --model "$MODEL" --prompt "$PROMPT" \
    --ctx-size "$CONTEXT" --max-new-tokens "$TOKENS" --seed "$SEED"
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum "$OUT/timeline.nsys-rep"
```

Choose report names from installed help. Preserve raw reports; for phase/stream queries export SQLite and inspect its schema. NVTX ranges exist only if emitted. Add narrow instrumentation when needed, disclose its overhead, and remove only your own temporary edits.

For individual decode kernels, repeat the focused diagnostic capture with `--cuda-graph-trace=node` when supported by installed help. Node tracing can add significant overhead; keep its timings separate from acceptance runs and the graph-level control. `--no-cuda-graphs` selects a different execution path and is not a substitute for observing nodes in graph replay.

## Critical-path attribution

- Join CPU submissions to GPU activities via correlation IDs, PID/TID, stream and graph IDs. Include generated **and native-library** kernels, copies, allocation and waits. Host API duration is not kernel duration.
- Within an explicit `[start,end]` phase clip intervals and compute their **union** for busy time. Separate compute/copy activity and overlap; include leading/trailing idle. Summed kernel durations can exceed elapsed time across streams; their shares are work shares, not wall-time shares. Busy and idle come from one capture and window. Idle is exposed GPU idle, not host overhead; explain it with the next bullet's evidence. Never subtract GPU busy time from another run's wall time: timing scope, overlap and execution conditions differ.
- Explain exposed gaps using dependencies and CPU runnable/scheduled/blocked state, JVM GC/safepoints/JIT, driver compilation or missing submissions. A long synchronization may wait for useful GPU work. Correlate Java evidence before blaming host scheduling.
- Attribute copies by buffer, direction, bytes and frequency, including native interop and observable migration. No memcpy does not prove no movement. Residency expectations come from the selected plan, not a universal rule about weights/KV.
- Prove graph on/off changes the path. Check capture granularity and node coverage before claiming kernels disappeared or handing a kernel to `jitllm-ncu-analysis`.

Return commands/artifact, phase window/coverage, total/count/per-call work, dependency evidence, hypotheses and discriminating test. Use `jitllm-ncu-analysis` for a consequential identified kernel; gaps alone do not justify kernel counters.

Reference: [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/).
