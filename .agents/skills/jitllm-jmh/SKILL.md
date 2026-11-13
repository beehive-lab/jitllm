---
name: jitllm-jmh
description: Design and interpret JMH microbenchmarks for isolated measured JitLLM Java hotspots. Use after host profiling, not for whole-inference or asynchronous GPU acceptance timing.
license: Apache-2.0
---

# Isolate a Java cost

Require an operation, realistic state/inputs and a host profile explaining relevance. Prefer existing benchmarks. Inspect `pom.xml` before assuming JMH exists; use an isolated module/source set if needed, preserving normal builds. Do not add permanent dependencies merely for an experiment.

- Match production sizes/distributions/lifetimes/contention. Choose `@State` scope and threads deliberately. Setup/teardown must not remove allocations/conversions required in production; reset mutable/recurrent state between cases.
- Return results or use `Blackhole`; both prevent dead-code elimination when used correctly. Vary runtime inputs to defeat folding/hoisting. Avoid large manual loops; when needed declare operation count and inspect generated code before inferring per-operation cost.
- Use warmup/measurement iterations and independent JVM forks; establish convergence instead of copying fixed counts. Execute the built JMH harness, not an IDE or ad hoc timer. Verify JDK/TornadoVM options reach forks; preserve JMH/JDK/GC/flags, parameters, CPU/power state, samples and JSON.
- Choose throughput/average/sample/single-shot for the question. Startup is not warmed throughput. Profile allocations/assembly in separate diagnostic passes. Investigate harness overhead for tiny operations without blindly subtracting noisy baselines.
- An async GPU enqueue benchmark measures enqueue unless completion is included. Label intentional dispatch measurements and separately inspect completion/overlap; do not claim kernel/end-to-end gains.
- Report effect size/uncertainty across forks. Overlapping confidence intervals do not prove equivalence; noisy results are inconclusive. Verify correctness then representative end-to-end performance using `jitllm-performance-verify`.

Read only relevant [official JMH samples](https://github.com/openjdk/jmh/tree/master/jmh-samples/src/main/java/org/openjdk/jmh/samples) for DCE, constant folding, loops, state or forking.
