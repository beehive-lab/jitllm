---
name: jitllm-ncu-analysis
description: >
  Nsight Compute workflow for one jitllm CUDA kernel. Use after nsys, the
  TornadoVM profiler or the metrics output has already named the hot kernel.
tags:
  - gpullama
  - ncu
  - kernel-profiling
license: Apache-2.0
---

# Nsight Compute on one jitllm kernel

## When to use

Only with a named target kernel, identified by `jitllm-nsys-analysis`, the TornadoVM
profiler, or generated-kernel inspection. Never run broad `ncu` collection over a whole
decode loop: replay serializes every launch and the run stops being the run you were
studying.

CUDA only.

## 1. Name the kernel

TornadoVM generates kernel names from task names, so the name in the timeline is the one to
match. Confirm it against the generated source rather than guessing:

```bash
./jitllm --gpu --model "<model.gguf>" --prompt "hi" -n 1 --print-kernel
```

Which kernel a run selects depends on device capability, not on backend name: warp-shuffle
GEMVs, 32-wide subgroup reductions, split-KV attention and tensor-core MMA are each gated,
and the non-gated sibling is a different kernel. Profiling one and attributing the result to
the other is the common mistake here.

## 2. Skip warm-up

The first launches carry JIT and first-touch allocation. Skip them explicitly, and profile a
bounded number of steady-state launches:

```bash
ncu --section SpeedOfLight --csv \
    --kernel-name regex:"<kernel>" \
    --launch-skip 32 --launch-count 16 \
    -- ./jitllm --gpu --model "<model.gguf>" \
         --prompt "<prompt>" -n 128 --seed 42
```

If the run reports `ERR_NVGPUCTRPERM`, the counters are restricted to admin users: `ncu`
attaches, the program still runs and produces correct output, and no metrics come back.
That is a machine configuration, not a command error — raise it with whoever owns the
host rather than working around it.

Pick `--launch-skip` from what the phase does: prefill launches once per layer per chunk,
decode once per layer per token, so a skip that clears warm-up in one phase does not in the
other. Use a command that reaches the phase you are studying — `--with-prefill-decode` and
`--batch-prefill-size` change which kernels exist at all.

## 3. Escalate from the classification

Start with `SpeedOfLight`, then add sections according to what it says:

| First reading | Add |
| --- | --- |
| high memory throughput, low compute | `MemoryWorkloadAnalysis` |
| high compute throughput, lower memory | `ComputeWorkloadAnalysis` |
| both low, short kernels or low occupancy | `LaunchStats`, `Occupancy` |
| occupancy fine, both still low | `WarpStateStats`, `SchedulerStats` |

For a transformer kernel, read in particular: DRAM throughput, L1 and L2 hit behaviour,
achieved occupancy, register count, shared and local memory use, global load width, and
instruction count.

## 4. Interpret against what this project can change

The kernels are Java methods compiled by TornadoVM. There is no hand-written PTX to tune, so
an actionable finding is one of:

- a different kernel variant already exists and the capability gate is choosing the wrong
  one;
- the worker grid or local work-group size is wrong for this shape;
- a buffer is in the wrong layout or the wrong representation for the access pattern;
- the kernel is doing work the program should not be asking for;
- the limit is a TornadoVM codegen property, which is an upstream report, not a local fix.

Never modify TornadoVM to make a measurement look better.

## 5. What to hand back

- the target kernel regex and why that kernel;
- the exact `ncu` command and the exact `jitllm` command;
- the raw relevant output, or the path to the exported metrics (outside the repository);
- the speed-of-light classification;
- occupancy, register and shared-memory observations;
- one recommended change, and which of the categories above it falls into.

Report measured numbers only. Do not present an estimate, and do not carry a number over
from another device, model or execution mode.
