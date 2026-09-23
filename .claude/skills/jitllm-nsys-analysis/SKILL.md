---
name: jitllm-nsys-analysis
description: >
  Nsight Systems workflow for jitllm on the CUDA backend. Use to capture a
  timeline and decide where the time goes: generated kernels, launch overhead,
  synchronization, host-device transfers, or GPU idle.
tags:
  - gpullama
  - nsys
  - profiling
license: Apache-2.0
---

# Nsight Systems on jitllm

## When to use

For the system-level picture: which kernels dominate, whether launch overhead is
significant, whether copies or synchronization are visible, and whether the GPU is idle
between kernels. For per-kernel occupancy, stalls and speed-of-light, use
`jitllm-ncu-analysis` after this identifies the hot kernel.

CUDA only. `nsys` sees nothing useful on the OpenCL or Metal backends.

## 1. Establish the run before profiling it

A profile of the wrong path is worse than no profile. Fix the tuple first: model file,
quantization, execution mode, and JDK/SDK line. Then confirm the run is what you think it
is, exactly as `build-n-run-engine` describes — resolved backend, an `execution_path` in
the metrics JSON, and correct generated text.

Choose flags that reach the phase you are studying:

| Studying | Flags |
| --- | --- |
| decode | a short prompt and a large `-n` |
| prefill | a long prompt and a small `-n` |
| sequential prefill | `--with-prefill-decode` |
| batched prefill | `--with-prefill-decode --batch-prefill-size N` (default-off otherwise) |
| launch overhead on CUDA | with and without `--cuda-graphs` |

## 2. Separate warm-up from steady state

The first tokens include GGUF load, host-to-device weight copy and TornadoVM JIT. Left in,
they dominate the timeline and hide the decode loop.

Two ways, in order of preference:

1. Give the run enough tokens that the steady state dominates, then restrict the reports
   to that range with `nsys stats --filter-time <start>/<end>`.
2. Run once with `--verbose-init` first, read the load and JIT timings, and subtract them
   when reading the totals.

Compilation should happen once. If kernel compilation appears repeatedly in the timeline,
that is a finding in itself: a compiled program is meant to be built once and invoked many
times.

## 3. Capture

Write traces to a timestamped directory outside the repository. Profiler output is never
committed.

```bash
OUT="$HOME/nsys-runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

nsys profile \
  --trace=cuda,nvtx,osrt \
  --output "$OUT/decode" \
  --force-overwrite true \
  ./jitllm --gpu --model "<model.gguf>" \
    --prompt "<prompt>" -n 256 --seed 42
```

Quote every path.

## 4. Report

```bash
nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum "$OUT/decode.nsys-rep"
nsys analyze "$OUT/decode.nsys-rep"
```

`nsys stats --help-reports` lists the report names for the installed version; `nsys analyze
--help` lists its rules. Check them rather than assuming — they change between releases.

## 5. Classify before recommending

| Observation | Classification | Next |
| --- | --- | --- |
| one generated kernel dominates GPU time | kernel-bound | `jitllm-ncu-analysis` on that kernel |
| many short kernels, high launch API time | launch-overhead-bound | compare with `--cuda-graphs`; look at graph count and per-layer task count |
| high `cudaMemcpy` time | transfer-bound | which buffers move per step? A weight or block table copied every execution is a residency question |
| high synchronization API time | sync-bound | look at the read-back at the invocation boundary |
| GPU gaps between kernels | host-bound scheduling | TornadoVM interpreter or host-side sampling |
| compilation appears after warm-up | a caching defect | the compiled program is being rebuilt |

Distinguish device-resident data from data that crosses the boundary each step. Weights,
the KV pool, the block table and the workspace stay on the device for the compiled
program's life; only values — token, position, block-table slot, sampling parameters —
should move per invocation. A per-step copy of anything else is the finding.

## 6. What to hand back

- the exact `nsys` command and the exact `jitllm` command;
- the trace path (outside the repo);
- top kernels by total GPU time, the CUDA API summary and the transfer summary;
- which part of the timeline is warm-up and which is steady state;
- the classification, and the single next action.

Every number must come from `nsys stats`, `nsys analyze` or a parsed trace. Do not estimate
and do not carry a number over from a different tuple.
