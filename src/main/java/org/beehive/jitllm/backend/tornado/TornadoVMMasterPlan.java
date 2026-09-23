package org.beehive.jitllm.backend.tornado;

import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.metrics.MetricsSink;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * Common contract for all TornadoVM GPU execution plans.
 *
 * <p>Three concrete implementations exist:
 *
 * <ul>
 *   <li>{@link TornadoVMMasterPlanSingleToken} — baseline single-token forward pass (preprocessing
 *       + N layers + logits).
 *   <li>{@link TornadoVMMasterPlanPrefillDecode} — sequential prefill/decode separation; reuses the
 *       same N layer graphs for both phases, skipping logits during prefill.
 *   <li>{@link TornadoVMMasterPlanBatchPrefillDecode} — batched prefill + single-token decode;
 *       holds 2N+3 graphs in one plan to keep the KV cache on device across phases.
 * </ul>
 *
 * <p>The {@link #initializeTornadoVMPlan} factory selects the implementation based on {@code
 * jitllm.withPrefillDecode} and {@code jitllm.prefillBatchSize}:
 *
 * <ul>
 *   <li>{@code withPrefillDecode=false} → {@link TornadoVMMasterPlanSingleToken}
 *   <li>{@code withPrefillDecode=true}, {@code prefillBatchSize=1} → {@link
 *       TornadoVMMasterPlanPrefillDecode}
 *   <li>{@code withPrefillDecode=true}, {@code prefillBatchSize>1} → {@link
 *       TornadoVMMasterPlanBatchPrefillDecode}
 * </ul>
 */
public interface TornadoVMMasterPlan {

    /**
     * The deprecated per-stage initialization log lines. {@code --verbose} reports the same facts
     * once, in the startup report, so it no longer enables these; only the legacy property does.
     */
    boolean ENABLE_TORNADOVM_INIT_TIME = Boolean.getBoolean("jitllm.EnableTimingForTornadoVMInit");

    /** When {@code true}, {@code withCUDAGraph()} is called — CUDA backend only. */
    boolean CUDA_GRAPHS = Boolean.parseBoolean(System.getProperty("jitllm.cudaGraphs", "false"));

    /**
     * @deprecated Replaced by {@code state.executionPolicy()}. This constant survives only for
     *     {@code State} and {@code Qwen2MoEState}, which use the batch size to <b>size arrays</b> —
     *     a capacity input, not policy — and for the bench harness. It is not read on any execution
     *     path.
     */
    @Deprecated boolean WITH_PREFILL_DECODE = Boolean.getBoolean("jitllm.withPrefillDecode");

    /**
     * @deprecated see {@link #WITH_PREFILL_DECODE}. Capacity input only.
     */
    @Deprecated int PREFILL_BATCH_SIZE = Integer.getInteger("jitllm.prefillBatchSize", 1);

    /**
     * Factory: creates, JIT-compiles, and warms up the appropriate TornadoVMMasterPlan.
     *
     * <p>When {@code jitllm.withPrefillDecode=true} and {@code jitllm.prefillBatchSize > 1}, a
     * {@link TornadoVMMasterPlanBatchPrefillDecode} is returned. Otherwise a {@link
     * TornadoVMMasterPlanSingleToken} is returned (used for the baseline path and the sequential
     * prefill/decode path when batch size is 1).
     *
     * @param state the model state
     * @param model the model instance
     * @return the initialized plan, also stored via {@link Model#setTornadoVMPlan}
     */
    static TornadoVMMasterPlan initializeTornadoVMPlan(State state, Model model) {
        return initializeTornadoVMPlan(state, model, MetricsSink.disabled());
    }

    /**
     * As {@link #initializeTornadoVMPlan(State, Model)}, with the plan reporting device-side
     * measurements to {@code sink}.
     *
     * <p>The sink is taken here rather than installed later because TornadoVM's profiler must be
     * switched on before the plan compiles and executes. With the default disabled sink the
     * profiler is never enabled and the reporting is one boolean test per execution.
     */
    static TornadoVMMasterPlan initializeTornadoVMPlan(State state, Model model, MetricsSink sink) {
        // Plan construction is where the device buffers are actually allocated, so it is where an
        // over-budget configuration fails — on the lowered path and the legacy one alike, which is
        // why the boundary is the whole method rather than the construction sites inside it.
        try {
            return buildPlan(state, model, sink);
        } catch (RuntimeException failure) {
            if (DeviceMemoryExhaustedException.isDeviceExhaustion(failure)) {
                throw DeviceMemoryExhaustedException.wrap(failure);
            }
            throw failure;
        }
    }

    private static TornadoVMMasterPlan buildPlan(State state, Model model, MetricsSink sink) {
        TornadoVMMasterPlan plan;

        // Every GPU plan passes here, the facade's and the harness's alike, so an FP16 cache the
        // selected layers do not implement is refused before any device buffer exists. The facade
        // has already asked at load; this covers the callers that build a state themselves.
        Fp16KeyValueSupport.require(model, state.executionPolicy(), state.storageOptions(), true);
        NativeLibrarySupport.require(model, state.executionPolicy(), true);

        // The lowering's opt-in is consulted here, in the one factory every caller reaches, rather
        // than at each construction site. It was branched at two sites before — the API session and
        // the golden harness — which is why the CLI, the server and the benchmark script silently
        // ran the legacy path however the flag was set: a paired A/B taken through the script was
        // measuring legacy against legacy. `handles` answers false unless the opt-in is set and the
        // tuple is the one the slice implements, so this costs a boolean read otherwise.
        if (org.beehive.jitllm.backend.tornado.lowering.LoweredPlanSelection.handles(
                model, state)) {
            reportPath(org.beehive.jitllm.runtime.backend.ExecutionPath.LOWERED, model, state);
            return org.beehive.jitllm.backend.tornado.lowering.LoweredPlanSelection.lower(
                    model, state, sink);
        }
        reportPath(org.beehive.jitllm.runtime.backend.ExecutionPath.LEGACY, model, state);

        // Resolved from the session's policy, once, here — not from a class constant read at
        // initialization.
        var policy = state.executionPolicy();
        boolean prefillDecode =
                policy.phaseStrategy()
                        == org.beehive.jitllm.runtime.policy.ExecutionPolicy.PhaseStrategy
                                .PREFILL_DECODE;
        if (prefillDecode && policy.prefillBatchSize() > 1) {
            // GPU path with batched prefill/decode
            plan = new TornadoVMMasterPlanBatchPrefillDecode(state, model, sink);
        } else if (prefillDecode) {
            // GPU path with simple prefill/decode
            plan = new TornadoVMMasterPlanPrefillDecode(state, model, sink);
        } else {
            // GPU path with no prefill/decode
            plan = new TornadoVMMasterPlanSingleToken(state, model, sink);
        }
        // Deliberately not stored on the model. A plan belongs to the session that built it;
        // parking it on the shared model meant the second session to start silently replaced the
        // first session's plan, and a later CPU call would then take the GPU branch against a
        // plan bound to someone else's buffers.
        return plan;
    }

    // @formatter:on

    /**
     * Creates the appropriate {@link TornadoExecutionPlan} instance for the given {@link Model} and
     * {@link State}.
     */
    /** Describes an already prepared plan without executing a token. */
    default org.beehive.jitllm.runtime.backend.ExecutionInfo executionInfo() {
        throw new UnsupportedOperationException("This plan does not expose execution diagnostics");
    }

    TornadoExecutionPlan createExecutionPlan();

    void forceCopyInReadOnlyData();

    /**
     * Returns the session to the sequence condition it was created in.
     *
     * <p>Nothing for a family whose only per-sequence memory is a key/value cache. For one with
     * recurrent state it zeroes that state <b>and puts the zeros on the device</b>: the buffers are
     * uploaded once and then persist there, so clearing the host arrays alone would leave the
     * accelerator conditioning the next sequence on the last one.
     */
    void resetSequenceState();

    /**
     * The shared implementation of {@link #resetSequenceState()}: zero on the host, then upload.
     *
     * <p>{@code transferToDevice} waits for the device before it returns, so the state is zero
     * there by the time the caller runs again.
     *
     * <p><b>Through one graph, not the whole plan.</b> A plan-wide upload runs once per task-graph
     * that takes the object as a parameter, and every layer graph takes this state: on the 27B's
     * batched plan that is 128 uploads of the same 151 MB buffer, 19 GB and 0.7 s for a reset that
     * has one buffer to write. The graphs share the device buffer — that is what {@code
     * consumeFromDevice} between them means — so one graph that binds it writes the copy all of
     * them read.
     *
     * @param graphBindingRecurrentState index of a graph that takes these buffers as parameters
     */
    static void resetSequenceState(
            TornadoExecutionPlan executionPlan, State state, int graphBindingRecurrentState) {
        state.resetSequenceState();
        Object[] recurrent = state.recurrentDeviceBuffers();
        if (recurrent.length > 0) {
            executionPlan.withGraph(graphBindingRecurrentState).transferToDevice(recurrent);
        }
    }

    FloatArray tornadoVMForwardDecode(int position);

    /** Releases all device memory held by this plan. */
    void freeTornadoExecutionPlan();

    /**
     * Records which path this session took, and the exact combination [D-7].
     *
     * <p>Reported <b>here</b>, in the one factory every caller reaches, for the same reason the
     * lowering opt-in is consulted here: it was branched at two sites once, and the CLI, the server
     * and the benchmark script all silently ran the legacy path however the flag was set. A report
     * emitted anywhere else would have the same hole.
     */
    private static void reportPath(
            org.beehive.jitllm.runtime.backend.ExecutionPath path,
            org.beehive.jitllm.model.Model model,
            org.beehive.jitllm.inference.state.State state) {
        var combination =
                org.beehive.jitllm.backend.tornado.lowering.LoweredPlanSelection.combinationOf(
                        model, state);
        boolean qualified =
                org.beehive.jitllm.backend.tornado.lowering.LoweringQualification.isQualified(
                        combination.architecture(), combination.dtype(), combination.mode());
        org.beehive.jitllm.auxiliary.RunMetrics.setExecutionPath(
                path.reportName(),
                combination.toString(),
                qualified,
                org.beehive.jitllm.backend.tornado.lowering.LoweredPlanSelection.mode()
                        .name()
                        .toLowerCase(java.util.Locale.ROOT));
    }

    /**
     * Reports the lowered path for a caller that acquires its lowered program itself.
     *
     * <p>{@link #buildPlan} is no longer "the one factory every caller reaches": the facade caches
     * compiled programs and calls {@code LoweredPlanSelection.lower} directly, so a session created
     * through it built a lowered plan and reported nothing. That stayed invisible while the CLI
     * went around the facade; once the CLI entered through it, every standalone CI row lost its
     * {@code execution_path}, and the assertion step reads that field to tell an accelerator run
     * from a CPU fallback.
     *
     * <p>Reported per session rather than per compile, because the cache answers a second session
     * without calling the supplier at all — and the question the field answers ("which path did
     * this run take") is a property of the run, not of the compile.
     */
    public static void reportLoweredPath(
            org.beehive.jitllm.model.Model model, org.beehive.jitllm.inference.state.State state) {
        reportPath(org.beehive.jitllm.runtime.backend.ExecutionPath.LOWERED, model, state);
    }
}
