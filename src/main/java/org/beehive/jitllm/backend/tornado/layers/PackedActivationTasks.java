package org.beehive.jitllm.backend.tornado.layers;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspace;
import org.beehive.jitllm.runtime.backend.DeviceCapability;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * The decode-path pieces every family shares when its projections run as packed integer dot
 * products ({@code dp4a}) over an activation quantized to Q8 blocks: when the path is taken, the
 * tasks that quantize the activation into the workspace's {@code wrapXbQuants / wrapXbScales /
 * wrapXbSums} triple, and the grids they and the projections run on.
 *
 * <p>The triple is one buffer set reused by every activation of a layer, so a projection reads
 * whatever the last quantize task of its own graph wrote; each family emits the quantize next to
 * the projection that reads it.
 */
// @formatter:on
public final class PackedActivationTasks {

    /**
     * Pins the floating-point projections where a family would otherwise take the packed ones. For
     * the tests whose subject is addressing rather than arithmetic: they compare the device against
     * the host exactly, which a quantized activation cannot do. Not a user option.
     */
    public static final String DISABLE_PROPERTY = "jitllm.qwen35.packedIntegerDot";

    private PackedActivationTasks() {}

    /** Whether the target device lowers {@code dp4a}. */
    public static boolean supported() {
        return TornadoDevices.target().capabilities().supports(DeviceCapability.PACKED_INTEGER_DOT);
    }

    /** {@link #supported()}, unless {@link #DISABLE_PROPERTY} is {@code false}. */
    public static boolean enabled() {
        return supported()
                && !"false".equalsIgnoreCase(System.getProperty(DISABLE_PROPERTY, "true"));
    }

    /**
     * The Q8 triple, for a layer graph to bind: uploaded once by the first layer graph and consumed
     * by name by the ones after it, so every graph addresses the same device buffers.
     */
    public static Object[] buffers(TornadoWorkspace ws) {
        return new Object[] {ws.wrapXbQuants, ws.wrapXbScales, ws.wrapXbSums};
    }

    /**
     * The RMS norm's apply into {@code ws.wrapXb}, quantized into the Q8 triple in the same task.
     * {@code temp[0]} is the scale the reduce wrote.
     */
    public static void rmsApplyAndQuantize(
            TaskGraph graph,
            String name,
            KernelContext context,
            TornadoWorkspace ws,
            FloatArray rmsWeights,
            FloatArray temp) {
        graph.task(
                name,
                TransformerComputeKernelsQ4_0::rmsApplyAndQuantizeActivationQ8Blocks,
                context,
                ws.wrapXb,
                ws.wrapX,
                rmsWeights,
                temp,
                ws.wrapXbQuants,
                ws.wrapXbScales,
                ws.wrapXbSums);
    }

    /** {@code activation} quantized into the Q8 triple. */
    public static void quantize(
            TaskGraph graph,
            String name,
            KernelContext context,
            TornadoWorkspace ws,
            FloatArray activation) {
        graph.task(
                name,
                TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                context,
                activation,
                ws.wrapXbQuants,
                ws.wrapXbScales,
                ws.wrapXbSums);
    }

    /** The grid of a quantize task: one 32-lane workgroup per block of 32 activations. */
    public static WorkerGrid quantizeWorker(int length) {
        return WorkerGridFactory.genericWorker(length, 32);
    }

    /**
     * The grid of a row-per-workgroup packed projection: {@code local} lanes, a multiple of 32, per
     * output row.
     */
    public static WorkerGrid projectionWorker(int rows, int local) {
        return WorkerGridFactory.genericWorker(rows * local, local);
    }
}
