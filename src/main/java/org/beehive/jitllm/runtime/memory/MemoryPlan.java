package org.beehive.jitllm.runtime.memory;

import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import org.beehive.jitllm.api.Experimental;

/**
 * What a configuration is predicted to need on the device, computed <b>before any allocation</b>.
 *
 * <h2>Three quantities, deliberately not merged</h2>
 *
 * <ul>
 *   <li>{@link #logicalBytes()} — the buffers themselves, shared storage counted once. What a model
 *       "is".
 *   <li>{@link #predictedBudgetBytes()} — what the backend's memory budget is predicted to be
 *       charged, including duplication and headers. <b>This is the number admission compares.</b>
 *   <li><b>Physical process or device memory</b> — deliberately <b>absent</b>. It is not part of
 *       this contract, and no method here reports it.
 * </ul>
 *
 * <p>The first two differ by more than rounding. Measured on {@code Llama-3.2-1B-Instruct-F16},
 * batched prefill's logical bytes are ~2.4 GiB and its budget consumption ~4.2 GiB, because the
 * per-layer weights are bound into two graph families and allocated twice. A preflight that
 * reported only logical bytes would admit that load against a 3 GiB budget and it would still die
 * part-allocated.
 */
@Experimental
public record MemoryPlan(
        List<MemoryComponent> components,
        long configuredBudgetBytes,
        Confidence confidence,
        String assumptions) {

    /**
     * How much the prediction can be relied on.
     *
     * <p>Present because "I do not know" must be representable. A plan that silently assumed a
     * multiplicity of 1 for an unrecognised topology would under-predict exactly where
     * under-predicting is most dangerous.
     */
    public enum Confidence {

        /**
         * Every component's multiplicity is known from the plan topology.
         *
         * <p><b>The only level that may enforce fail-fast admission.</b> A load whose exact plan
         * exceeds the configured budget is refused before allocation.
         */
        EXACT,

        /**
         * At least one component used a conservative upper bound rather than a known value.
         *
         * <p><b>Report-only.</b> Never rejects a load on its own — a caller must opt in explicitly
         * for that. An upper bound that happens to exceed the budget would block a load that would
         * in fact have run, and the caller has no way to overrule a refusal from inside the loader.
         */
        CONSERVATIVE,

        /**
         * The topology is not recognised.
         *
         * <p><b>No admission decision may be made from this plan</b>, in either direction. It does
         * not say a load will fit and it does not say it will not.
         */
        UNSUPPORTED
    }

    public MemoryPlan {
        components = List.copyOf(components);
    }

    /** The buffers themselves, shared storage counted once, duplication excluded. */
    public long logicalBytes() {
        return components.stream().mapToLong(MemoryComponent::logicalBytes).sum();
    }

    /** What the backend's budget is predicted to be charged. The admission quantity. */
    public long predictedBudgetBytes() {
        return components.stream().mapToLong(MemoryComponent::predictedBytes).sum();
    }

    /** Bytes attributable to buffers being allocated more than once. */
    public long duplicationBytes() {
        return components.stream().mapToLong(MemoryComponent::duplicationBytes).sum();
    }

    /** Backend-required headers and alignment. */
    public long overheadBytes() {
        return components.stream().mapToLong(MemoryComponent::overheadBytes).sum();
    }

    /**
     * Whether the <b>prediction</b> fits the configured budget.
     *
     * <p><b>Not a guarantee that a load will succeed</b>, at any confidence level. It compares two
     * numbers: what this plan predicts, and what the backend was configured to allow. It knows
     * nothing about what else is resident on the device, what the driver reserves, or what another
     * process is doing — none of which is in this contract.
     *
     * <p>At {@link Confidence#CONSERVATIVE} a {@code false} may be pessimistic, and at {@link
     * Confidence#UNSUPPORTED} the answer carries no weight in either direction. Only an {@link
     * Confidence#EXACT} plan is used to refuse a load.
     */
    public boolean fitsConfiguredBudget() {
        return configuredBudgetBytes <= 0 || predictedBudgetBytes() <= configuredBudgetBytes;
    }

    /** Components largest first — what a person needs in order to act on a shortfall. */
    public List<MemoryComponent> dominantComponents() {
        return components.stream()
                .sorted(Comparator.comparingLong(MemoryComponent::predictedBytes).reversed())
                .toList();
    }

    /** The largest single component, when there is one. */
    public Optional<MemoryComponent> largestComponent() {
        return dominantComponents().stream().findFirst();
    }

    /** A report a person can read, and the text an over-capacity failure carries. */
    public String describe() {
        StringBuilder out = new StringBuilder();
        out.append("Device memory plan (predicted before allocation)\n");
        out.append("  assumptions: ").append(assumptions).append('\n');
        out.append("  confidence:  ").append(confidence).append('\n');
        out.append(
                String.format("  %-26s %14s %6s %14s%n", "component", "logical", "x", "predicted"));
        for (MemoryComponent c : dominantComponents()) {
            out.append(
                    String.format(
                            "  %-26s %11s %6d %11s%n",
                            c.name(),
                            mib(c.logicalBytes()),
                            c.multiplicity(),
                            mib(c.predictedBytes())));
        }
        out.append(
                String.format(
                        "  %-26s %11s %6s %11s%n",
                        "TOTAL", mib(logicalBytes()), "", mib(predictedBudgetBytes())));
        out.append("    of which duplication: ")
                .append(mib(duplicationBytes()))
                .append(", headers/alignment: ")
                .append(mib(overheadBytes()))
                .append('\n');
        if (configuredBudgetBytes > 0) {
            out.append("  configured budget: ")
                    .append(mib(configuredBudgetBytes))
                    .append(
                            switch (confidence) {
                                case EXACT ->
                                        fitsConfiguredBudget()
                                                ? "  (predicted to fit)"
                                                : "  (predicted NOT to fit)";
                                case CONSERVATIVE ->
                                        fitsConfiguredBudget()
                                                ? "  (a conservative estimate fits; not a guarantee)"
                                                : "  (a conservative estimate does not fit; may be pessimistic)";
                                case UNSUPPORTED ->
                                        "  (topology not recognised; no judgement offered)";
                            })
                    .append('\n');
        }
        out.append(
                "  physical free device memory is not part of this contract and is not reported");
        return out.toString();
    }

    private static String mib(long bytes) {
        return String.format("%.1f MiB", bytes / 1048576.0);
    }
}
