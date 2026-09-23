package org.beehive.jitllm.backend.tornado.lowering;

import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.jitllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.architecture.ArchitectureInputs;
import org.beehive.jitllm.model.architecture.ModelArchitecture;
import org.beehive.jitllm.model.architecture.ModelArchitectures;
import org.beehive.jitllm.program.InferenceProgram;
import org.beehive.jitllm.runtime.backend.CompileOptions;
import org.beehive.jitllm.runtime.backend.DeviceCapabilities;
import org.beehive.jitllm.runtime.backend.DeviceCapability;
import org.beehive.jitllm.runtime.metrics.MetricsSink;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy.SamplingResidency;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * The one internal branch that decides whether a session's plan comes from the lowering or from the
 * legacy path.
 *
 * <h2>The legacy path is the default</h2>
 *
 * <p>Off unless {@code jitllm.lowering} is set, and applicable to exactly one tuple: <b>Llama, FP16,
 * single-token</b>. Everything else — other families, {@code Q8_0}, the prefill/decode and
 * batch-prefill/decode modes — takes the path it took before, unchanged. Those modes stay green as
 * non-regression checks rather than being claimed by this slice (acceptance, corrected).
 *
 * <p>The flag is transitional and internal. It is not public API, and it goes away when the tuple's
 * acceptance is green and the lowered path becomes the default for it.
 */
public final class LoweredPlanSelection {

    /**
     * The lowering input: {@code auto} (default), {@code on} or {@code off} [D-6].
     *
     * <p>Was a boolean opt-in; {@code true}/{@code false} still parse, as {@code on}/{@code off}.
     * The name is unchanged so no existing script or invocation breaks.
     */
    public static final String ENABLE_PROPERTY = "jitllm.lowering";

    /**
     * Providers used to declare their supported modes independently, and two declared all three
     * while this branch rejected everything but {@code STANDARD}. That is an inconsistent
     * capability contract: a matrix built from the declarations would claim six prefill-mode
     * combinations that have never executed a lowered graph. {@code
     * LoweringCapabilityConsistencyTest} now holds the two in agreement against this set.
     */
    public static final java.util.Set<ExecutionMode> SELECTABLE_MODES =
            java.util.EnumSet.of(ExecutionMode.STANDARD);

    /**
     * How many times the lowered path has actually produced a plan.
     *
     * <p>The only sound way for a test to assert the lowering ran. Reading the property back, or
     * grepping a log, proves that something was <i>asked for</i> — not that it <i>happened</i>.
     * That distinction is not academic: a run of this slice's accel gate looked green while the
     * flag had never reached the JVM at all, because a {@code -DargLine} override replaced the
     * profile's own arguments. A test that requests the lowered path must fail when execution falls
     * back, and this counter is what lets it.
     */
    private static final java.util.concurrent.atomic.AtomicLong LOWERED_PLANS =
            new java.util.concurrent.atomic.AtomicLong();

    private LoweredPlanSelection() {}

    /** Whether the lowered path is switched on. */
    public static boolean enabled() {
        return mode() != LoweringMode.OFF;
    }

    /**
     * The requested mode.
     *
     * <p><b>Read per call, not folded at class initialization</b>, for the reason above: a {@code
     * static final} read would be constant-folded before any test could set it.
     */
    public static LoweringMode mode() {
        return LoweringMode.parse(System.getProperty(ENABLE_PROPERTY), LoweringMode.AUTO);
    }

    /** How many plans the lowered path has produced. Test observation, not a metric. */
    public static long loweredPlanCount() {
        return LOWERED_PLANS.get();
    }

    /**
     * Whether this exact tuple is the one the slice implements.
     *
     * <p>Deliberately narrow. Broadening it is a follow-up slice with its own acceptance, not a
     * condition to relax here.
     */
    public static boolean handles(Model model, State state) {
        LoweringMode mode = mode();
        if (mode == LoweringMode.OFF) {
            return false;
        }
        boolean implemented = implemented(model, state);
        if (mode == LoweringMode.ON) {
            if (!implemented) {
                throw new UnsupportedLoweringException(combinationOf(model, state));
            }
            return true;
        }
        // AUTO: proven combinations only. Everything else selects legacy deliberately — configured
        // behaviour, not a failure, so it is silent [D-4].
        return implemented
                && LoweringQualification.isQualified(
                        model.architectureId(), weightRepresentation(model), executionMode(state));
    }

    /**
     * Whether a session of this model, under this policy, can take the lowered path: {@link
     * #handles} answered before any state exists.
     *
     * <p>What a model allocates up front for the lowering — the shared key/value pool, sized for
     * several sessions — has to follow this answer and not {@link #enabled()}. {@code auto} is
     * enabled for every model yet selects legacy for all but the qualified set, and a legacy
     * session given the pool pays for sessions it never shares with.
     */
    public static boolean mayHandle(Model model, ExecutionPolicy policy) {
        return switch (mode()) {
            case OFF -> false;
                // ON fails loudly in handles() for an unimplemented combination; it needs the pool.
            case ON -> true;
            case AUTO ->
                    implemented(model, policy)
                            && LoweringQualification.isQualified(
                                    model.architectureId(),
                                    weightRepresentation(model),
                                    executionMode(policy));
        };
    }

    /**
     * {@link #mayHandle} for a model whose weights are not loaded, as a preflight has it: the
     * architecture, and the loaded weight type when the file determines it.
     *
     * @param weightType the materialized weight type, or {@code null} when unknown — then any type
     *     is assumed, so the answer can only err toward the lowered path and the shared pool that
     *     comes with it, over-counting a reservation rather than missing one
     */
    public static boolean mayHandle(
            org.beehive.jitllm.runtime.model.ArchitectureId architecture,
            DataType weightType,
            ExecutionPolicy policy) {
        return switch (mode()) {
            case OFF -> false;
            case ON -> true;
            case AUTO ->
                    LoweringQualification.qualified().stream()
                            .anyMatch(
                                    c ->
                                            c.architecture().equals(architecture)
                                                    && (weightType == null
                                                            || c.dtype() == weightType)
                                                    && c.mode() == executionMode(policy));
        };
    }

    /** The exact triple qualification is keyed on, for messages and metrics. */
    public static LoweringQualification.Combination combinationOf(Model model, State state) {
        return new LoweringQualification.Combination(
                model.architectureId(), weightRepresentation(model), executionMode(state));
    }

    /** Whether a lowered implementation exists for this exact combination and could run it. */
    public static boolean implemented(Model model, State state) {
        return implemented(model, state.executionPolicy());
    }

    private static boolean implemented(Model model, ExecutionPolicy policy) {
        // Device-resident sampling is not implemented. Its token id lives in a domain-owned array,
        // and reading it after the lock is released is exactly the escape the invocation boundary
        // exists to prevent; carrying it out properly is follow-up work.
        if (policy.samplingResidency() == SamplingResidency.DEVICE) {
            return false;
        }
        if (!SELECTABLE_MODES.contains(executionMode(policy))) {
            // Single-token only: the prefill/decode topologies keep the legacy plan.
            return false;
        }
        if (!ProgramShape.SUPPORTED_WEIGHTS.contains(weightRepresentation(model))) {
            return false;
        }
        // Two questions, two owners: does anything describe this architecture, and
        // can this backend run the triple.
        if (!ModelArchitectures.isDescribed(model.architectureId(), ARCHITECTURES.get())) {
            return false;
        }
        return TornadoBackendSupport.supports(
                model.architectureId(), weightRepresentation(model), executionMode(policy));
    }

    /**
     * Describes this model's forward pass as a program.
     *
     * @throws IllegalArgumentException if this family has no description yet — callers that must
     *     not fail ask {@link #handles} first, which is what the selection branch does
     */
    public static InferenceProgram describe(
            Model model, ExecutionPolicy policy, DataType keyValueRepresentation) {
        InferenceProgram program = describeIfSupported(model, policy, keyValueRepresentation);
        if (program == null) {
            throw new IllegalArgumentException(
                    "no program description for "
                            + model.configuration().getClass().getSimpleName()
                            + "; the lowered path handles"
                            + " Llama and Qwen2 so far, and every other family stays on the legacy plan");
        }
        return program;
    }

    /**
     * The description for this model, or {@code null} when no architecture computes it.
     *
     * <p>A missing architecture is a {@code null} here rather than a throw: four of the ten
     * families this project loads have no description, and the legacy path is correct for them.
     */
    private static InferenceProgram describeIfSupported(
            Model model, ExecutionPolicy policy, DataType keyValueRepresentation) {
        var id = model.architectureId();
        var architectures = ARCHITECTURES.get();
        if (!ModelArchitectures.isDescribed(id, architectures)) {
            return null;
        }
        return ModelArchitectures.select(id, architectures)
                .describe(
                        new ArchitectureInputs(
                                model.configuration(),
                                weightRepresentation(model),
                                keyValueRepresentation,
                                policy));
    }

    /**
     * Discovered once. {@code ServiceLoader} walks the classpath, and a decode loop asking that
     * question per session would be paying for a lookup whose answer cannot change.
     */
    private static final java.util.function.Supplier<java.util.List<ModelArchitecture>>
            ARCHITECTURES =
                    new java.util.function.Supplier<>() {
                        private java.util.List<ModelArchitecture> cached;

                        @Override
                        public synchronized java.util.List<ModelArchitecture> get() {
                            if (cached == null) {
                                cached = ModelArchitectures.discover();
                            }
                            return cached;
                        }
                    };

    /**
     * How this model's weights are represented <b>on the device</b>.
     *
     * <p>Not what the file held: a Q4_0 or K-quant file is materialized as {@link DataType#Q8_0},
     * and it is the materialized form the task graphs are built for and the signature must carry.
     */
    private static DataType weightRepresentation(Model model) {
        return model.weights().dataType();
    }

    /** The cache key for this model, in this binding domain. */
    public static ProgramCacheKey key(
            Model model,
            BindingDomain domain,
            ExecutionPolicy policy,
            DataType keyValueRepresentation) {
        var device = org.beehive.jitllm.backend.tornado.device.TornadoDevices.current();
        return ProgramCacheKey.of(
                describe(model, policy, keyValueRepresentation).signature(),
                device.backend(),
                device.id(),
                compileOptions(),
                capabilities(model),
                domain);
    }

    /**
     * The lowering for this model, resolved through backend registration.
     *
     * <p>It was a chain of configuration-type checks that had to be kept in step with the
     * description chain beside it. The two questions are now asked of the two components that can
     * answer them: what the model computes, of the architecture; what this backend can run, of the
     * backend.
     */
    private static FamilyLowering loweringFor(Model model, ExecutionMode mode) {
        return TornadoBackendSupport.lowering(
                model.architectureId(),
                weightRepresentation(model),
                mode,
                compileOptions(),
                capabilities(model));
    }

    /** Which execution mode a state's policy asks for, in the backend's vocabulary. */
    private static ExecutionMode executionMode(State state) {
        return executionMode(state.executionPolicy());
    }

    private static ExecutionMode executionMode(ExecutionPolicy policy) {
        if (policy.phaseStrategy() != ExecutionPolicy.PhaseStrategy.PREFILL_DECODE) {
            return ExecutionMode.STANDARD;
        }
        return policy.prefillBatchSize() > 1
                ? ExecutionMode.BATCH_PREFILL_DECODE
                : ExecutionMode.PREFILL_DECODE;
    }

    /** Validates the program and lowers it, or throws naming what did not match. */
    public static TornadoVMMasterPlan lower(Model model, State state, MetricsSink sink) {
        TornadoVMMasterPlan plan =
                loweringFor(model, executionMode(state))
                        .lower(
                                describe(
                                        model,
                                        state.executionPolicy(),
                                        keyValueRepresentation(state)),
                                state,
                                model,
                                sink);
        LOWERED_PLANS.incrementAndGet();
        return plan;
    }

    static CompileOptions compileOptions() {
        return new CompileOptions(TornadoVMMasterPlan.CUDA_GRAPHS);
    }

    /**
     * What lowering may vary on for this model.
     *
     * <p>The scheduler type stays a model-and-device question — Mistral is {@code NON_NVIDIA}
     * regardless of hardware — which is why this takes a model and {@code
     * TornadoDevices.current().capabilities()} is not used directly here.
     */
    static DeviceCapabilities capabilities(Model model) {
        return SchedulerDetectionService.determineSchedulerType(model) == SchedulerType.NVIDIA
                ? DeviceCapabilities.of(DeviceCapability.SINGLE_PASS_RMS)
                : DeviceCapabilities.NONE;
    }

    /**
     * How this workspace's key/value entries are stored.
     *
     * <p>Part of the signature, not a lowering choice: it changes the dtype of the fixed key/value
     * bindings even though the component sequence is identical (review, 13.2).
     */
    private static DataType keyValueRepresentation(State state) {
        return state.usesFp16KeyValueCache() ? DataType.F16 : DataType.F32;
    }

    /**
     * A stable identifier for the device a program is compiled for.
     *
     * <p><b>Not {@code System.getProperty("tornado.device")}.</b> TornadoVM sets that property
     * while initialising, so it read {@code "default"} before the first plan was built and {@code
     * "0"} afterwards — which meant the first session and every later one produced different cache
     * keys for the same program. Two sessions happened to share because both ran after
     * initialisation; four did not, and the session sweep is what exposed it.
     *
     * <p>The string is unchanged: the platform name, or {@code "unavailable"} where there is no
     * accelerator. That matters, because it is a cache key component and a different label would
     * silently invalidate nothing while keying everything differently.
     */
    private static String deviceLabel() {
        return org.beehive.jitllm.backend.tornado.device.TornadoDevices.current().displayName();
    }
}
