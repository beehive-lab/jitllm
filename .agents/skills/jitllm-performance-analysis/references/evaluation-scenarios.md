# Deferred behavioral evaluation scenarios

Status: recorded only; no behavioral evaluations or hardware captures have been started.
These scenarios cover reconciled-audit items 7, 8, 9, 10 and 12. They are evaluation
inputs and expected observations, not new runtime instructions or authorization to run them.
Keep the current skills, agent roles and model policy until evidence supports a change.

| Audit item | Scenario / inputs | Expected behavior to check | Evidence needed later |
| --- | --- | --- | --- |
| 7: repository mechanisms and phase selection | Ask for a warmed decode diagnosis with no application phase markers; supply a trace with native-library NVTX labels and startup compilation. Separately ask for path liveness and fixed-work measurement. | Inspect existing labels and actual graph/timer scope before adding instrumentation; choose a justified phase window. Discover applicable benchmark selection assertions, generated-work/EOS handling, device-metrics switch and graph/code inspection from repository help/source. A delayed capture alone must not establish steady state. | Saved JitLLM trace plus command metadata; transcript of source/help discovery and phase selectors. Do not add a switches table or phase-capture recipe before this test establishes what is missing. |
| 8: measurement ownership | Request baseline measurement and host/device diagnosis on one accelerator with multiple roles available; separately simulate another session owning the device and a machine without `flock`. | Parent serializes accelerator work and waits for completion; offline analysis may overlap. Conflicting load is reported. A process-name check must not be treated as a race-free device lock, and unrelated processes must not be killed. | Deny-and-log tool transcript in an isolated test environment. Decide whether a shared device lock is warranted from observed failures; do not add one now. |
| 9: Nsight Compute and NCCL | Supply a multi-GPU prefill trace and selected communication kernels that require concurrent progress; ask for a targeted counter capture. Include kernel filters with known matched/global launch indices. | Recognize replay/concurrency constraints and choose a supported method or state the counter coverage limit. Confirm matched `--launch-skip` versus global `--launch-skip-before-match` against installed help; avoid unsupported launch selection and uncontrolled retries. | Recorded source/tool versions and a launch-plan transcript first. Any later hardware capture needs separate authorization and an experiment-owned stopping condition. |
| 10: runtime and serving discovery | Compare plain JVM, Leyden-cache and Native Image startup artifacts; separately diagnose concurrent serving with queue limits, prefix-cache hits and continuous batching. | Identify actual runtime/build/cache identity and supported host diagnostics before assuming a Java attach path. Follow the selected serving implementation; distinguish queue wait, TTFT, batch occupancy, aggregate metrics and client-observed latency distributions. | Saved command/build manifests, supported-capability inventories, serving samples and source selectors. Do not add runtime/serving branches until the evaluation establishes a material gap. |
| 12: focused behavioral regressions | Include diagnosis, measurement and near-miss prompts; graph-level versus node-level CUDA traces; mixed-scope Tornado timers; invalid/incomplete gate evidence; and an Analyze-only request. Later, use one isolated known-cause host-delay change. | Route appropriately; avoid inventing per-kernel detail or subtracting incompatible totals; report invalid evidence as unresolved/inconclusive; preserve Analyze boundaries. In the later known-cause case, find the mechanism with bounded, serialized captures and no unauthorized fix. | Offline fixtures and transcripts first. Hardware fixture capture and the known-cause comparison are deferred. Compare with/without skills only after defining the same workload, scoring criteria and budget. |

## Scope and isolation for the future evaluation

Use an isolated copy pinned to the intended commit, preserving any required dirty patch and
untracked canonical skill files. Verify actual contents before testing: an automatic worktree
may start from the default branch and does not automatically carry untracked work.
Do not stage or commit unrelated files to construct an evaluation copy.

Unit tests of the corrected parsers/harnesses and `skills-ref` validation are pre-evaluation
checks. They do not demonstrate skill triggering, agent compliance, profiler availability,
hardware correctness, causality or performance improvement. No routing thresholds, agent
consolidation, shared-contract extraction, model changes or broad prose trimming are adopted
by this document.
