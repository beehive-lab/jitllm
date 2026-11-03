package org.beehive.jitllm.backend.tornado.plan;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.backend.tornado.plan.prepacking.WeightPrepacking;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspace;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/**
 * Model weights read in place from the mapped GGUF file, on a device that shares memory with the
 * host, instead of copied into device buffers ({@code TaskGraph.consumeFromUnifiedMemory}).
 *
 * <p>Example: on an Apple M4 Pro, Qwen3-8B Q8_0 keeps its 8.7 GB of weights once, as file pages,
 * rather than once in the mapping and once more in Metal buffers.
 *
 * <p>On where the device shares memory with the host, unless declined with {@code
 * -Djitllm.weights.unifiedMemory=false} or MLX is asked for ({@code -Djitllm.nativeLibraries=true}:
 * MLX reads host-made copies), and only for a family whose graphs all bind their weights through
 * here ({@link
 * TornadoPlanProvider#unifiedMemoryWeights}): TornadoVM refuses a plan in which one graph reads an
 * array in place and another uploads it. Only the GGUF file's mapped tensors are read in place; a
 * prepacked projection keeps its upload, since the prepacking graph writes its packed copy on the
 * device.
 */
public final class UnifiedMemoryWeights {

    /** The property that declines it: {@code -Djitllm.weights.unifiedMemory=false}. */
    public static final String PROPERTY = "jitllm.weights.unifiedMemory";

    private UnifiedMemoryWeights() {}

    /** Whether a plan of {@code family} reads its weights in place. */
    public static boolean applies(TornadoPlanProvider family) {
        return !"false".equals(System.getProperty(PROPERTY))
                && !mlxAskedFor()
                && family.unifiedMemoryWeights()
                && deviceSupports();
    }

    /** Whether {@code -Djitllm.nativeLibraries=true} asks for MLX, which reads its own copies. */
    private static boolean mlxAskedFor() {
        return ExecutionPolicy.nativeLibrariesSpecified()
                && ExecutionPolicy.fromSystemProperties().nativeLibraries();
    }

    /** Whether the device plans run on shares memory with the host, as TornadoVM sees it. */
    public static boolean deviceSupports() {
        try {
            return device().isUnifiedMemorySupported();
        } catch (RuntimeException | LinkageError e) {
            return false;
        }
    }

    /**
     * Declares in {@code graph}, read in place, those of {@code weights} the device can read so,
     * and returns the others for the caller to bind as before. Returns {@code weights} unchanged
     * when the plan does not read weights in place.
     */
    public static Object[] consume(TaskGraph graph, TornadoWorkspace ws, Object... weights) {
        return consume(graph, ws.unifiedMemory, ws.prepacking, weights);
    }

    /**
     * {@link #consume(TaskGraph, TornadoWorkspace, Object...)} for a caller that holds the plan's
     * answer and prepacking rather than the workspace: the {@code WeightPrepacker}.
     */
    public static Object[] consume(
            TaskGraph graph, boolean readsInPlace, WeightPrepacking prepacking, Object... weights) {
        if (!readsInPlace) {
            return weights;
        }
        TornadoDevice device = device();
        Map<Object, Boolean> inPlace = new IdentityHashMap<>();
        List<Object> others = new ArrayList<>();
        for (Object w : weights) {
            Object array = w instanceof TornadoTensor t ? t.deviceArray() : w;
            // Only the GGUF file's own mapped tensors: a workspace buffer the device writes may be
            // page-aligned too, and TornadoVM checks the storage, not the accesses.
            if (array instanceof TornadoNativeArray mapped
                    && mapped.getSegment().isMapped()
                    && !prepacking.prepackedArray(array)
                    && device.isOnUnifiedMemory(array)) {
                inPlace.put(array, Boolean.TRUE);
            } else {
                others.add(w);
            }
        }
        if (!inPlace.isEmpty()) {
            graph.consumeFromUnifiedMemory(inPlace.keySet().toArray());
        }
        return others.toArray();
    }

    /** {@link #consume}, then the others uploaded once; returns those uploaded. */
    public static Object[] upload(TaskGraph graph, TornadoWorkspace ws, Object... weights) {
        Object[] others = consume(graph, ws, weights);
        if (others.length > 0) {
            graph.transferToDevice(DataTransferMode.FIRST_EXECUTION, others);
        }
        return others;
    }

    private static TornadoDevice device() {
        return TornadoRuntimeProvider.getTornadoRuntime().getDefaultDevice();
    }
}
