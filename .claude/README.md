# Claude Agents And Skills For jitllm

This directory contains project-local Claude Code agents and skills for jitllm work, including building, benchmarking, profiling, model porting, and framework integrations.

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
| `jitllm-benchmarking-specialist` | Reproducible benchmarks: baseline/target comparisons, backend matrices, metrics JSON, before/after reports |
| `jitllm-perf-profiling-specialist` | Root-causing performance with the TornadoVM profiler, nsys and ncu once a benchmark shows a regression or target; bottleneck classification |

## Skill Boundaries

| Skill | Use For |
|---|---|
| `build-tornado` | Building a TornadoVM SDK (7.3 or newer) for the chosen JDK line and backend |
| `build-n-run-engine` | Building jitllm with Maven and running it on an accelerator |
| `jitllm-nsys-analysis` | Nsight Systems timeline on CUDA: generated kernels, launch overhead, sync, transfers, GPU idle |
| `jitllm-ncu-analysis` | Nsight Compute analysis of one already-identified hot CUDA kernel |
| `port-model-to-jitllm` | Porting a new model architecture, or deciding whether an existing family can be reused |
| `update-langchain4j-integration` | Updating the LangChain4j jitllm integration to a new jitllm release |
| `update-quarkus-langchain4j-integration` | Updating the Quarkus LangChain4j jitllm extension to a new jitllm release |
| `update-spring-ai-integration` | Creating or updating the Spring AI jitllm integration for a jitllm release |

## Operating Rules

- Do not invent performance numbers. Every claim must come from metrics JSON, TornadoVM profiler output, system profilers, kernel profilers, or generated kernel evidence.
- Use `nsys` before `ncu`: identify hot kernels and launch/sync behavior first, then profile a specific kernel.
- Keep artifacts under timestamped directories, preferably `perf-results/<timestamp>/` or a clearly named scratch directory.
- For codegen-sensitive work, inspect generated kernel source and add tests that fail if emission regresses.
- Match the backend and profiler to the task. CUDA-specific work should use CUDA evidence; backend comparisons should measure each requested backend under equivalent conditions.
