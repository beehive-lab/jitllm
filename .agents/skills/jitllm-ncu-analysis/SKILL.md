---
name: jitllm-ncu-analysis
description: Diagnose one identified JitLLM CUDA kernel using Nsight Compute counters, generated code and launch geometry after evidence establishes its phase and end-to-end relevance.
license: Apache-2.0
---

# CUDA kernel diagnosis

Require kernel, phase, shape, representative launch and impact evidence; otherwise use `jitllm-nsys-analysis`. Generated TornadoVM and cuBLAS/cuDNN kernels have different owners/remedies.

Record `ncu --version`, `--help`, `--list-sections` and supported metrics. Match exact variant to task/plan, device capability, dtype, shape, grid/block and graph mode. Inspect selected generated code where available; a name may combine different shapes/instances.

Capture a few representative launches, not a generation. Choose skip/count from observed launches at the intended prefill/decode/KV-depth window; verify whether skip counts matched or global launches in this version. Record replay mode, graph support, cache/clock controls and resulting perturbation.

```bash
ncu --section SpeedOfLight --kernel-name "regex:$KERNEL_REGEX" \
  --launch-skip "$SKIP" --launch-count "$COUNT" --export "$OUT/kernel" \
  ./jitllm --gpu --cuda --model "$MODEL" --prompt "$PROMPT" \
    --ctx-size "$CONTEXT" --max-new-tokens "$TOKENS" --seed "$SEED"
```

For `ERR_NVGPUCTRPERM`, report restricted coverage and continue timeline/source analysis; do not silently change security settings. Replay can serialize streams/change caches. Its elapsed runtime is not an acceptance benchmark.

| Hypothesis | Useful installed sections |
| --- | --- |
| Traffic/cache/access efficiency | `MemoryWorkloadAnalysis` |
| Instruction mix/pipeline throughput | `ComputeWorkloadAnalysis` |
| Small grid/register/shared-memory residency | `LaunchStats`, `Occupancy` |
| Dependency/latency/issue stalls | `SchedulerStats`, `WarpStateStats` |

Use measured bytes, cache levels, achieved bandwidth, arithmetic intensity and relevant instruction throughput together. Low occupancy or one dominant kernel alone is not proof; more occupancy can harm register/cache behavior. Label modeled rooflines and byte/FLOP assumptions. Do not assume every token reads all stored weights or a universal percentage of peak bandwidth.

Map evidence to variant selection, worker grid, workload/layout/fusion, native-library algorithm/interop, or TornadoVM lowering/codegen. Locate only responsible methods under `src/main/java/org/beehive/jitllm/backend/tornado/{kernels,layers,plan,ops/nativelib}`. Compiler claims need emitted-code evidence; native hotspots usually need shape/layout/algorithm analysis before replacement.

Return target identity/instances, exact command/artifact, counters/units, supported mechanism and alternative, critical-path relevance and candidate test. Use `jitllm-performance-verify` for correctness and unprofiled end-to-end proof.

Reference: [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).
