package org.beehive.jllm.runtime.policy;

import java.util.Objects;
import java.util.OptionalInt;
import org.beehive.jllm.api.Experimental;

/**
 * How a session executes: the choices that select <b>which components run and how</b>.
 *
 * <p>An immutable value with structural equality, <b>resolved once when a session is created</b>
 * and never re-read per token. That is the whole point of it. Today these choices are {@code static
 * final} fields initialized from system properties at class initialization, which has three
 * consequences this type removes: a property set after the class loads silently does nothing, two
 * sessions of one model cannot differ, and tests reach for orderings and workarounds to control
 * something that should have been a parameter.
 *
 * <h2>What belongs here, and what does not</h2>
 *
 * <p>The line is not "settings a user might want to change". It is:
 *
 * <blockquote>
 *
 * A value that determines the <b>type, layout, count or capacity of a device array</b> is not
 * policy. A value that selects <b>which components run against arrays already sized</b> is.
 *
 * </blockquote>
 *
 * <p>So the phase strategy, the working prefill batch size, where sampling happens and the working
 * split-KV partition count are policy — each selects among components bound to buffers that already
 * exist. Context capacity, the key/value representation, block size and count, and the sizing
 * <i>maxima</i> those working values must fit inside are <b>not</b>: they belong to whoever
 * allocates.
 *
 * <p>A policy is part of a program's signature, and therefore of the compiled-program cache key:
 * two sessions with different policies resolve to different compiled programs rather than one
 * program behaving differently.
 *
 * <h2>Where it lives</h2>
 *
 * <p>In {@code runtime}, not in the facade, for the same reason the metrics seam is there: it is
 * <b>written from above</b> — a caller sets it through {@code ModelOptions} or {@code
 * SessionOptions} — and <b>read from below</b>, by the plans and layers that must not depend on the
 * public API package. A policy type in {@code api} would have made every backend that reads a
 * policy depend on the facade.
 *
 * @see Overrides for the field-by-field session override
 */
@Experimental
public final class ExecutionPolicy {

    /** Whether a turn runs as one token at a time, or as a prefill followed by decode. */
    public enum PhaseStrategy {
        SINGLE_TOKEN,
        PREFILL_DECODE
    }

    /** Where the next token is chosen. */
    public enum SamplingResidency {
        /** The logits come back and the host samples. */
        HOST,
        /** The device chooses, and the full logits row never leaves it. */
        DEVICE
    }

    private final PhaseStrategy phaseStrategy;
    private final int prefillBatchSize;
    private final SamplingResidency samplingResidency;
    private final OptionalInt splitKvPartitions;
    private final boolean packedHalf2Attention;
    private final boolean scalarFp16KeyValueReads;
    private final boolean nativeLibraries;

    private ExecutionPolicy(
            PhaseStrategy phaseStrategy,
            int prefillBatchSize,
            SamplingResidency samplingResidency,
            OptionalInt splitKvPartitions,
            boolean packedHalf2Attention,
            boolean scalarFp16KeyValueReads,
            boolean nativeLibraries) {
        this.nativeLibraries = nativeLibraries;
        this.packedHalf2Attention = packedHalf2Attention;
        this.scalarFp16KeyValueReads = scalarFp16KeyValueReads;
        this.phaseStrategy = Objects.requireNonNull(phaseStrategy, "phaseStrategy");
        this.prefillBatchSize = prefillBatchSize;
        this.samplingResidency = Objects.requireNonNull(samplingResidency, "samplingResidency");
        this.splitKvPartitions = Objects.requireNonNull(splitKvPartitions, "splitKvPartitions");
        if (prefillBatchSize < 1) {
            throw new IllegalArgumentException(
                    "prefillBatchSize must be at least 1: " + prefillBatchSize);
        }
        if (splitKvPartitions.isPresent() && splitKvPartitions.getAsInt() < 1) {
            throw new IllegalArgumentException(
                    "splitKvPartitions must be at least 1 when set: "
                            + splitKvPartitions.getAsInt());
        }
        if (phaseStrategy == PhaseStrategy.SINGLE_TOKEN && prefillBatchSize != 1) {
            throw new IllegalArgumentException(
                    "a single-token policy has no prefill batch;"
                            + " prefillBatchSize was "
                            + prefillBatchSize
                            + ". Select PREFILL_DECODE, or leave the batch at 1 — a policy that names a"
                            + " batch it will not use would read as a setting that had been applied.");
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    /** A builder seeded from an existing policy, for a field-by-field change. */
    public static Builder from(ExecutionPolicy base) {
        Objects.requireNonNull(base, "base");
        return new Builder()
                .phaseStrategy(base.phaseStrategy)
                .prefillBatchSize(base.prefillBatchSize)
                .samplingResidency(base.samplingResidency)
                .splitKvPartitions(base.splitKvPartitions)
                .packedHalf2Attention(base.packedHalf2Attention)
                .scalarFp16KeyValueReads(base.scalarFp16KeyValueReads)
                .nativeLibraries(base.nativeLibraries);
    }

    /**
     * The defaults this build runs with, read from the {@code llama.*} system properties.
     *
     * <p><b>One place</b>, and read per call rather than folded into a constant — which is the
     * defect being removed, so reproducing it here would be self-defeating. Callers resolve a
     * policy once per session and carry the value; nothing consults this in a loop.
     */
    public static ExecutionPolicy fromSystemProperties() {
        boolean prefillDecode = Boolean.getBoolean("jllm.withPrefillDecode");
        int prefillBatch = Integer.getInteger("jllm.prefillBatchSize", 1);
        return builder()
                .phaseStrategy(
                        prefillDecode ? PhaseStrategy.PREFILL_DECODE : PhaseStrategy.SINGLE_TOKEN)
                .prefillBatchSize(prefillDecode ? Math.max(1, prefillBatch) : 1)
                .samplingResidency(
                        Boolean.getBoolean("jllm.deviceSample")
                                ? SamplingResidency.DEVICE
                                : SamplingResidency.HOST)
                .splitKvPartitions(
                        Boolean.getBoolean("jllm.attention.splitKv")
                                ? OptionalInt.of(
                                        Integer.getInteger("jllm.attention.splitKv.count", 8))
                                : OptionalInt.empty())
                .packedHalf2Attention(Boolean.getBoolean("jllm.attention.deepHalf2"))
                .scalarFp16KeyValueReads(Boolean.getBoolean("jllm.kvcache.fp16.scalar"))
                .nativeLibraries(nativeLibrariesFromProperties())
                .build();
    }

    /** The property that asks for vendor native libraries: {@code -Djllm.nativeLibraries=true}. */
    public static final String NATIVE_LIBRARIES_PROPERTY = "jllm.nativeLibraries";

    /**
     * The earlier spelling, from when the native prefill libraries were on by default. Honoured
     * when set; the default is now off.
     */
    public static final String LEGACY_NATIVE_PREFILL_PROPERTY = "jllm.prefill.native";

    private static boolean nativeLibrariesFromProperties() {
        String current = System.getProperty(NATIVE_LIBRARIES_PROPERTY);
        String legacy = System.getProperty(LEGACY_NATIVE_PREFILL_PROPERTY);
        if (current != null
                && legacy != null
                && Boolean.parseBoolean(current) != Boolean.parseBoolean(legacy)) {
            throw new IllegalArgumentException(
                    "-D"
                            + NATIVE_LIBRARIES_PROPERTY
                            + "="
                            + current
                            + " and -D"
                            + LEGACY_NATIVE_PREFILL_PROPERTY
                            + "="
                            + legacy
                            + " disagree; drop "
                            + LEGACY_NATIVE_PREFILL_PROPERTY);
        }
        return Boolean.parseBoolean(current != null ? current : legacy);
    }

    public PhaseStrategy phaseStrategy() {
        return phaseStrategy;
    }

    /** Working prefill batch size; always 1 under {@link PhaseStrategy#SINGLE_TOKEN}. */
    public int prefillBatchSize() {
        return prefillBatchSize;
    }

    public SamplingResidency samplingResidency() {
        return samplingResidency;
    }

    /** Working split-KV partition count, or empty for the backend's own choice. */
    public OptionalInt splitKvPartitions() {
        return splitKvPartitions;
    }

    /** Whether the FP16 key/value scores accumulate packed, one {@code __hfma2} per pair. */
    public boolean packedHalf2Attention() {
        return packedHalf2Attention;
    }

    /**
     * Whether the FP16 key/value cache is read with scalar half loads instead of packed ones.
     *
     * <p>An evaluation aid, and a kernel variant on the same reasoning as {@link
     * #packedHalf2Attention()}.
     */
    public boolean scalarFp16KeyValueReads() {
        return scalarFp16KeyValueReads;
    }

    /**
     * Whether vendor native libraries (cuBLAS, cuDNN) may replace JIT kernels where an
     * implementation exists. Off by default: the JIT kernels run every supported configuration, and
     * the native path is experimental, narrower, and needs more device memory (stacked projection
     * copies beside the weights).
     */
    public boolean nativeLibraries() {
        return nativeLibraries;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof ExecutionPolicy that)) {
            return false;
        }
        return prefillBatchSize == that.prefillBatchSize
                && phaseStrategy == that.phaseStrategy
                && samplingResidency == that.samplingResidency
                && splitKvPartitions.equals(that.splitKvPartitions)
                && packedHalf2Attention == that.packedHalf2Attention
                && scalarFp16KeyValueReads == that.scalarFp16KeyValueReads
                && nativeLibraries == that.nativeLibraries;
    }

    @Override
    public int hashCode() {
        return Objects.hash(
                phaseStrategy,
                prefillBatchSize,
                samplingResidency,
                splitKvPartitions,
                packedHalf2Attention,
                scalarFp16KeyValueReads,
                nativeLibraries);
    }

    @Override
    public String toString() {
        return "ExecutionPolicy[phase="
                + phaseStrategy
                + ", prefillBatch="
                + prefillBatchSize
                + ", sampling="
                + samplingResidency
                + ", splitKv="
                + (splitKvPartitions.isPresent() ? splitKvPartitions.getAsInt() : "backend")
                + ", packedHalf2="
                + packedHalf2Attention
                + ", scalarFp16Kv="
                + scalarFp16KeyValueReads
                + ", nativeLibraries="
                + nativeLibraries
                + "]";
    }

    public static final class Builder {

        private PhaseStrategy phaseStrategy = PhaseStrategy.SINGLE_TOKEN;
        private int prefillBatchSize = 1;
        private SamplingResidency samplingResidency = SamplingResidency.HOST;
        private OptionalInt splitKvPartitions = OptionalInt.empty();
        private boolean packedHalf2Attention;
        private boolean scalarFp16KeyValueReads;
        private boolean nativeLibraries;

        private Builder() {}

        /** Allow vendor native libraries where implemented; experimental, off by default. */
        public Builder nativeLibraries(boolean nativeLibraries) {
            this.nativeLibraries = nativeLibraries;
            return this;
        }

        public Builder packedHalf2Attention(boolean packedHalf2Attention) {
            this.packedHalf2Attention = packedHalf2Attention;
            return this;
        }

        public Builder scalarFp16KeyValueReads(boolean scalarFp16KeyValueReads) {
            this.scalarFp16KeyValueReads = scalarFp16KeyValueReads;
            return this;
        }

        public Builder phaseStrategy(PhaseStrategy phaseStrategy) {
            this.phaseStrategy = Objects.requireNonNull(phaseStrategy, "phaseStrategy");
            return this;
        }

        public Builder prefillBatchSize(int prefillBatchSize) {
            this.prefillBatchSize = prefillBatchSize;
            return this;
        }

        public Builder samplingResidency(SamplingResidency samplingResidency) {
            this.samplingResidency = Objects.requireNonNull(samplingResidency, "samplingResidency");
            return this;
        }

        public Builder splitKvPartitions(OptionalInt splitKvPartitions) {
            this.splitKvPartitions = Objects.requireNonNull(splitKvPartitions, "splitKvPartitions");
            return this;
        }

        public Builder splitKvPartitions(int splitKvPartitions) {
            return splitKvPartitions(OptionalInt.of(splitKvPartitions));
        }

        public ExecutionPolicy build() {
            return new ExecutionPolicy(
                    phaseStrategy,
                    prefillBatchSize,
                    samplingResidency,
                    splitKvPartitions,
                    packedHalf2Attention,
                    scalarFp16KeyValueReads,
                    nativeLibraries);
        }
    }

    /**
     * A session's field-by-field override of the model's policy.
     *
     * <p>Separate from {@link ExecutionPolicy} because "override two fields" and "here is a whole
     * policy" are different statements, and a caller building a session usually does not know the
     * model's default to build from. An unset field means <b>keep the model's</b>, which a complete
     * policy cannot express.
     */
    public static final class Overrides {

        private static final Overrides NONE = builder().build();

        private final PhaseStrategy phaseStrategy;
        private final Integer prefillBatchSize;
        private final SamplingResidency samplingResidency;
        private final OptionalInt splitKvPartitions;
        private final boolean splitKvSet;
        private final Boolean nativeLibraries;

        private Overrides(Builder builder) {
            this.nativeLibraries = builder.nativeLibraries;
            this.phaseStrategy = builder.phaseStrategy;
            this.prefillBatchSize = builder.prefillBatchSize;
            this.samplingResidency = builder.samplingResidency;
            this.splitKvPartitions = builder.splitKvPartitions;
            this.splitKvSet = builder.splitKvSet;
        }

        /** Overrides nothing: the session runs the model's policy. */
        public static Overrides none() {
            return NONE;
        }

        public static Builder builder() {
            return new Builder();
        }

        /** Whether this overrides anything at all. */
        public boolean isEmpty() {
            return phaseStrategy == null
                    && prefillBatchSize == null
                    && samplingResidency == null
                    && !splitKvSet
                    && nativeLibraries == null;
        }

        /** The model's policy with this session's fields applied. */
        public ExecutionPolicy applyTo(ExecutionPolicy base) {
            Objects.requireNonNull(base, "base");
            if (isEmpty()) {
                return base;
            }
            ExecutionPolicy.Builder builder = ExecutionPolicy.from(base);
            if (phaseStrategy != null) {
                builder.phaseStrategy(phaseStrategy);
                // A strategy change without a batch change would otherwise inherit the base's
                // batch, and SINGLE_TOKEN with a batch greater than one is rejected at
                // construction. Falling back to 1 keeps the override meaning what it says.
                if (phaseStrategy == PhaseStrategy.SINGLE_TOKEN && prefillBatchSize == null) {
                    builder.prefillBatchSize(1);
                }
            }
            if (prefillBatchSize != null) {
                builder.prefillBatchSize(prefillBatchSize);
            }
            if (samplingResidency != null) {
                builder.samplingResidency(samplingResidency);
            }
            if (splitKvSet) {
                builder.splitKvPartitions(splitKvPartitions);
            }
            if (nativeLibraries != null) {
                builder.nativeLibraries(nativeLibraries);
            }
            return builder.build();
        }

        @Override
        public String toString() {
            return isEmpty()
                    ? "ExecutionPolicy.Overrides[none]"
                    : "ExecutionPolicy.Overrides[phase="
                            + phaseStrategy
                            + ", prefillBatch="
                            + prefillBatchSize
                            + ", sampling="
                            + samplingResidency
                            + ", splitKv="
                            + (splitKvSet ? String.valueOf(splitKvPartitions) : "unset")
                            + ", nativeLibraries="
                            + nativeLibraries
                            + "]";
        }

        public static final class Builder {

            private PhaseStrategy phaseStrategy;
            private Integer prefillBatchSize;
            private SamplingResidency samplingResidency;
            private OptionalInt splitKvPartitions = OptionalInt.empty();
            private boolean splitKvSet;
            private Boolean nativeLibraries;

            private Builder() {}

            /** Allow or forbid vendor native libraries for this session. */
            public Builder nativeLibraries(boolean nativeLibraries) {
                this.nativeLibraries = nativeLibraries;
                return this;
            }

            public Builder phaseStrategy(PhaseStrategy phaseStrategy) {
                this.phaseStrategy = Objects.requireNonNull(phaseStrategy, "phaseStrategy");
                return this;
            }

            public Builder prefillBatchSize(int prefillBatchSize) {
                this.prefillBatchSize = prefillBatchSize;
                return this;
            }

            public Builder samplingResidency(SamplingResidency samplingResidency) {
                this.samplingResidency =
                        Objects.requireNonNull(samplingResidency, "samplingResidency");
                return this;
            }

            /**
             * Sets the working partition count; {@code OptionalInt.empty()} means the backend's.
             */
            public Builder splitKvPartitions(OptionalInt splitKvPartitions) {
                this.splitKvPartitions =
                        Objects.requireNonNull(splitKvPartitions, "splitKvPartitions");
                this.splitKvSet = true;
                return this;
            }

            public Builder splitKvPartitions(int splitKvPartitions) {
                return splitKvPartitions(OptionalInt.of(splitKvPartitions));
            }

            public Overrides build() {
                return new Overrides(this);
            }
        }
    }
}
