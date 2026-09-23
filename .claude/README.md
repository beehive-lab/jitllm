# Claude Agents And Skills For jitllm

This directory contains project-local Claude Code agents and skills for jitllm work, including benchmarking, profiling, correctness debugging, and TornadoVM integration.

The scope is broad enough to support future repo work:

- jitllm benchmarking and profiling
- TornadoVM backend/codegen investigation
- correctness debugging for precision and kernel changes
- accelerator feature validation

After adding or editing files here, restart Claude Code from the repository root and run `/agents` and `/skills` to verify discovery.

## Portability Rules

- Do not hardcode user-specific paths. Discover the repository root from the current working directory and external project roots from environment variables, local config, documentation, or explicit user input.
- Do not bake current experiment settings into durable guidance. Prompt text, model paths, token limits, memory limits, feature flags, and backend choices come from the user request, checked-in scripts, or benchmark manifests.
- Keep current performance conclusions in result artifacts, not in reusable agent policy. Agents should collect evidence before classifying bottlenecks.
- Prefer commands that run from the repository root or use named environment variables such as `TORNADOVM_HOME` and `MODEL_DIR`.
- When local machine details are needed, record them in the benchmark artifact directory rather than in `.claude`.

## Agent Boundaries

| Agent | Use For |
|---|---|
| `jitllm-perf-profiling-specialist` | Running jitllm benchmarks, collecting metrics, nsys/ncu profiling, bottleneck classification |
| `tornado-backend-specialist` | TornadoVM backend changes, generated kernel codegen, native accelerator library integration |
| `jitllm-correctness-debug-agent` | Output/accuracy regressions after TornadoVM or jitllm performance changes |

## Skill Boundaries

| Skill | Use For |
|---|---|
| `jitllm-benchmarking` | Reproducible jitllm benchmark runs using local scripts |
| `jitllm-nsys-analysis` | Nsight Systems trace collection and system-level analysis |
| `jitllm-ncu-analysis` | Nsight Compute analysis of a specific hot CUDA kernel |
| `tornado-codegen-validation` | TornadoVM API/codegen/correctness validation for generated accelerator kernels |

## Operating Rules

- Do not invent performance numbers. Every claim must come from metrics JSON, TornadoVM profiler output, system profilers, kernel profilers, or generated kernel evidence.
- Use `nsys` before `ncu`: identify hot kernels and launch/sync behavior first, then profile a specific kernel.
- Keep artifacts under timestamped directories, preferably `perf-results/<timestamp>/` or a clearly named scratch directory.
- For codegen-sensitive work, inspect generated kernel source and add tests that fail if emission regresses.
- Match the backend and profiler to the task. CUDA-specific work should use CUDA evidence; backend comparisons should measure each requested backend under equivalent conditions.
