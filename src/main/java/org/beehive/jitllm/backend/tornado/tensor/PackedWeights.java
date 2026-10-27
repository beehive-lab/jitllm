package org.beehive.jitllm.backend.tornado.tensor;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.plan.ExecutionMode;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.backend.DeviceCapability;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.runtime.tensor.DataType;

// @formatter:off
/**
 * Whether a plan holds a representation in the packed tile layout ({@code TileLayout}), for a
 * family whose kernels read it as tiles. On by default wherever every kernel such a plan can choose
 * reads the tiles:
 *
 * <ul>
 *   <li>the device is a CUDA one with packed-integer dot products and int8 tensor cores;
 *   <li>a batched prefill runs at a width the int8 GEMM tiles (1, or a multiple of 128).
 * </ul>
 *
 * <p>{@code -Djitllm.q8.packed} and {@code -Djitllm.q4.packed} decide instead when set, either way.
 * A kernel that reads Q8_0 or Q4_0 blocks must never be handed tiles, so the paths without a packed
 * kernel refuse a tiled weight by name ({@link #noPackedKernel}).
 */
// @formatter:on
public final class PackedWeights {

    /** Tiles Q8_0 projections ({@code true}) or never ({@code false}); unset: the default above. */
    public static final String Q8_PROPERTY = "jitllm.q8.packed";

    /** Tiles Q4_0 projections ({@code true}) or never ({@code false}); unset: the default above. */
    public static final String Q4_PROPERTY = "jitllm.q4.packed";

    /** The int8 GEMM's row tile, which a batched prefill's width must fill. */
    private static final int GEMM_ROWS = 128;

    private PackedWeights() {}

    /**
     * Whether a family's packed-integer kernels are on: its {@code property} is not {@code false}.
     */
    public static boolean packedIntegerDotOn(String property) {
        return !"false".equalsIgnoreCase(System.getProperty(property, "true"));
    }

    /**
     * Whether a plan under {@code policy} holds {@code type} in tiles, for a family whose kernels
     * read it as tiles: {@code -Djitllm.q8.packed} or {@code -Djitllm.q4.packed} when set,
     * otherwise a CUDA device with the packed kernels' capabilities and a prefill width they tile.
     */
    public static boolean tiles(DataType type, ExecutionPolicy policy) {
        String property =
                switch (type) {
                    case Q8_0 -> Q8_PROPERTY;
                    case Q4_0 -> Q4_PROPERTY;
                    default -> null;
                };
        if (property == null) {
            return false;
        }
        String asked = System.getProperty(property);
        if (asked != null && !asked.isBlank()) {
            return Boolean.parseBoolean(asked);
        }
        return deviceRunsPackedKernels() && prefillWidthTiles(policy);
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
    static boolean prefillWidthTiles(ExecutionPolicy policy) {
        return ExecutionMode.of(policy) != ExecutionMode.BATCH_PREFILL_DECODE
                || policy.prefillBatchSize() % GEMM_ROWS == 0;
    }

    /** The refusal for a path that has no kernel over packed tiles. */
    public static UnsupportedOperationException noPackedKernel(String where) {
        return new UnsupportedOperationException(
                where
                        + " reads a packed projection, and this path has no kernel for the packed"
                        + " layout. Run with -Djitllm.q8.packed=false -Djitllm.q4.packed=false.");
    }
}
