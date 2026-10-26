package org.beehive.jitllm.backend.tornado.tensor;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.backend.DeviceCapability;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;

// @formatter:off
/**
 * Whether a load packs its projections ({@link PackedTiles}): every Q8_0 projection whose shape
 * fits, and every retained Q4_0 one. On by default wherever the model supports it, so a model runs
 * packed exactly when every kernel its plan can choose reads the packed layout:
 *
 * <ul>
 *   <li>the family has packed kernels on all its paths, which its loader states (Gemma 4's cuBLAS
 *       projections under native libraries have none, nor a family whose packed-integer kernels are
 *       switched off);
 *   <li>the device is a CUDA one with packed-integer dot products and int8 tensor cores;
 *   <li>a batched prefill runs at a width the int8 GEMM tiles (1, or a multiple of 128).
 * </ul>
 *
 * <p>The policy is the one the model is loaded for: the caller's, scoped to the load with {@link
 * #loading}, or the process's properties for a load outside one (the CLI's). A session that later
 * asks for a policy the packed layout cannot serve is refused ({@link #unsupported}).
 *
 * <p>{@code -Djitllm.q8.packed} and {@code -Djitllm.q4.packed} decide instead when set, either way.
 * A packed projection's tensor says so ({@link TornadoTensor#isPacked()}), and every kernel choice
 * consults it: a kernel that reads Q8_0 or Q4_0 blocks must never be handed packed tiles, so the
 * paths without a packed kernel refuse one by name ({@link #noPackedKernel}).
 */
// @formatter:on
public final class PackedWeights {

    /** Packs Q8_0 projections ({@code true}) or never ({@code false}); unset: the default above. */
    public static final String Q8_PROPERTY = "jitllm.q8.packed";

    /**
     * Packs retained Q4_0 projections ({@code true}) or never ({@code false}); unset: the default.
     */
    public static final String Q4_PROPERTY = "jitllm.q4.packed";

    /** The int8 GEMM's row tile, which a batched prefill's width must fill. */
    private static final int GEMM_ROWS = 128;

    /** The policy of the load running on this thread, if a caller scoped one. */
    private static final ThreadLocal<ExecutionPolicy> LOAD = new ThreadLocal<>();

    private PackedWeights() {}

    /**
     * Runs {@code load} with {@link #loadPolicy()} answering {@code policy}: the representation is
     * chosen for the policy the model is loaded with, not for the process's properties, and a load
     * on another thread is not affected.
     */
    public static <T, E extends Exception> T loading(ExecutionPolicy policy, Load<T, E> load)
            throws E {
        var previous = LOAD.get();
        LOAD.set(policy);
        try {
            return load.run();
        } finally {
            if (previous == null) {
                LOAD.remove();
            } else {
                LOAD.set(previous);
            }
        }
    }

    /** A load that {@link #loading} scopes. */
    @FunctionalInterface
    public interface Load<T, E extends Exception> {
        T run() throws E;
    }

    /** The policy the current load chooses representations for. */
    public static ExecutionPolicy loadPolicy() {
        var scoped = LOAD.get();
        return scoped != null ? scoped : ExecutionPolicy.fromSystemProperties();
    }

    /**
     * Why packed projections cannot run under {@code policy}, or empty when they can: a batched
     * prefill whose width the int8 GEMM, the only packed prefill kernel, does not tile.
     */
    public static java.util.Optional<String> unsupported(ExecutionPolicy policy) {
        return prefillWidthTiles(policy)
                ? java.util.Optional.empty()
                : java.util.Optional.of(
                        "its packed projections need a batched-prefill width of a multiple of "
                                + GEMM_ROWS
                                + ", not "
                                + policy.prefillBatchSize());
    }

    /**
     * Whether this load packs its Q8_0 projections.
     *
     * @param familySupports whether every path of the family reads packed Q8_0 tiles
     */
    public static boolean packQ8(boolean familySupports) {
        return decide(Q8_PROPERTY, familySupports);
    }

    /**
     * Whether this load packs its retained Q4_0 projections.
     *
     * @param familySupports whether every path of the family reads packed Q4_0 tiles
     */
    public static boolean packQ4(boolean familySupports) {
        return decide(Q4_PROPERTY, familySupports);
    }

    /**
     * Whether a family's packed-integer kernels are on: its {@code property} is not {@code false}.
     */
    public static boolean packedIntegerDotOn(String property) {
        return !"false".equalsIgnoreCase(System.getProperty(property, "true"));
    }

    private static boolean decide(String property, boolean familySupports) {
        String asked = System.getProperty(property);
        if (asked != null && !asked.isBlank()) {
            return Boolean.parseBoolean(asked);
        }
        return familySupports && deviceRunsPackedKernels() && prefillWidthTiles(loadPolicy());
    }

    /** CUDA, with the packed-integer dot products and int8 tensor cores the packed kernels use. */
    private static boolean deviceRunsPackedKernels() {
        var device = TornadoDevices.current();
        var capabilities = device.capabilities();
        return BackendId.CUDA.equals(device.backend())
                && capabilities.supports(DeviceCapability.PACKED_INTEGER_DOT)
                && capabilities.supports(DeviceCapability.INT8_TENSOR_CORE_MMA);
    }

    /**
     * A batched prefill's width fills the int8 GEMM's row tiles, the only packed prefill kernel.
     */
    private static boolean prefillWidthTiles(ExecutionPolicy policy) {
        if (policy.phaseStrategy() != ExecutionPolicy.PhaseStrategy.PREFILL_DECODE) {
            return true;
        }
        int width = policy.prefillBatchSize();
        return width == 1 || width % GEMM_ROWS == 0;
    }

    /** The refusal for a path that has no kernel over packed tiles. */
    public static UnsupportedOperationException noPackedKernel(String where) {
        return new UnsupportedOperationException(
                where
                        + " reads a packed projection, and this path has no kernel for the packed"
                        + " layout. Run with -Djitllm.q8.packed=false -Djitllm.q4.packed=false.");
    }
}
