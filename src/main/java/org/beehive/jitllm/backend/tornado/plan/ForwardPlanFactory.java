package org.beehive.jllm.backend.tornado.plan;

import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.runtime.tensor.DataType;

// @formatter:off
/**
 * Factory for {@link ForwardPlan} instances.
 *
 * <p>Dispatches across three axes in order:
 *
 * <ol>
 *   <li>Quantization ({@link DataType} — the representation the weights were materialized in)
 *   <li>Model family ({@link org.beehive.jllm.model.ModelType})
 *   <li>Execution mode ({@link ExecutionMode})
 * </ol>
 *
 * <p>Use the typed convenience methods when the execution mode is known at the call site:
 *
 * <ul>
 *   <li>{@link #createSingleToken} — returns {@link SingleTokenForwardPlan}
 *   <li>{@link #createPrefillDecode} — returns {@link PrefillDecodeForwardPlan}
 *   <li>{@link #createBatchPrefillDecode} — returns {@link BatchPrefillDecodeForwardPlan}
 * </ul>
 */
// @formatter:on
public class ForwardPlanFactory {

    private ForwardPlanFactory() {}

    // ── Typed public API ──────────────────────────────────────────────────────

    public static SingleTokenForwardPlan createSingleToken(
            DataType quantization, State state, Model model) {
        ForwardPlan plan = create(quantization, ExecutionMode.STANDARD, state, model);
        if (plan instanceof SingleTokenForwardPlan singleToken) return singleToken;
        throw new IllegalStateException(
                "Expected SingleTokenForwardPlan for STANDARD mode but got "
                        + plan.getClass().getSimpleName());
    }

    public static PrefillDecodeForwardPlan createPrefillDecode(
            DataType quantization, State state, Model model) {
        ForwardPlan plan = create(quantization, ExecutionMode.PREFILL_DECODE, state, model);
        if (plan instanceof PrefillDecodeForwardPlan prefillDecode) return prefillDecode;
        throw new IllegalStateException(
                "Expected PrefillDecodeForwardPlan for PREFILL_DECODE mode but got "
                        + plan.getClass().getSimpleName());
    }

    public static BatchPrefillDecodeForwardPlan createBatchPrefillDecode(
            DataType quantization, State state, Model model) {
        ForwardPlan plan = create(quantization, ExecutionMode.BATCH_PREFILL_DECODE, state, model);
        if (plan instanceof BatchPrefillDecodeForwardPlan batchPrefillDecode)
            return batchPrefillDecode;
        throw new IllegalStateException(
                "Expected BatchPrefillDecodeForwardPlan for BATCH_PREFILL_DECODE mode but got "
                        + plan.getClass().getSimpleName());
    }

    // ── Generic dispatch ──────────────────────────────────────────────────────

    /**
     * Dispatches on the representation the weights were <b>materialized</b> in, not on what the
     * file held. The two differ for every quantization the device has no kernel for: a Q4_K or Q4_0
     * file arrives here as {@link DataType#Q8_0}, which is exactly the plan it needs.
     *
     * <p>The format-decoded types are unreachable rather than unimplemented — nothing materializes
     * a device tensor in them — so they are refused by name instead of promising a plan that could
     * not be built.
     */
    static ForwardPlan create(DataType quantization, ExecutionMode mode, State state, Model model) {
        // The representation errors come first, before any provider is consulted: they are facts
        // about materialization, true for every architecture, and asking a provider about a dtype
        // that never reaches the device would scatter one answer across ten files.
        switch (quantization) {
            case F16, Q8_0, Q4_K, Q4_0 -> {}
            case F32 -> throw new UnsupportedOperationException("F32 plans not yet implemented");
            case Q4_1, Q5_K, Q6_K ->
                    throw new UnsupportedOperationException(
                            quantization
                                    + " is decoded during compute and is never materialized on the"
                                    + " device; it should have been mapped to Q8_0 at load");
            case BF16 ->
                    throw new UnsupportedOperationException(
                            "BF16 is narrowed to F16 when materialized for the device; it should have"
                                    + " been mapped to F16 at load");
        }

        // Absence is now a configuration error rather than a fallback: there is nothing left to
        // fall back to, and a model whose provider is missing would otherwise fail somewhere less
        // informative.
        return TornadoPlanRegistry.create(quantization, mode, state, model)
                .orElseThrow(
                        () ->
                                new UnsupportedOperationException(
                                        "no plan provider is registered for architecture '"
                                                + model.architectureId()
                                                + "' ("
                                                + model.getModelType()
                                                + "). "
                                                + "Registered: "
                                                + TornadoPlanRegistry.registeredNames()
                                                + ". A provider is a file plus a service entry; if this build was"
                                                + " shaded, check that META-INF/services survived."));
    }
}
