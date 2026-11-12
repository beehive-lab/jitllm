package org.beehive.jitllm.backend.tornado;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.runtime.backend.BackendId;

/**
 * Whether plans are captured and replayed as CUDA graphs: on by default on the CUDA backend, off
 * with {@code -Djitllm.cudaGraphs=false} ({@code --no-cuda-graphs}), and never on for OpenCL, Metal
 * or a host-only run, which have no CUDA graphs.
 */
final class CudaGraphDefault {

    /** The property {@code --cuda-graphs} and {@code --no-cuda-graphs} set. */
    static final String PROPERTY = "jitllm.cudaGraphs";

    private CudaGraphDefault() {}

    /** The setting for this process: the property, if set, on the target device's backend. */
    static boolean enabled() {
        boolean cuda;
        try {
            cuda = BackendId.CUDA.equals(TornadoDevices.target().backend());
        } catch (RuntimeException | LinkageError e) {
            cuda = false;
        }
        return enabled(System.getProperty(PROPERTY), cuda);
    }

    /**
     * The setting for a property value ({@code null} when unset) on a CUDA or other backend.
     * Package-private for the unit test.
     */
    static boolean enabled(String property, boolean cuda) {
        return cuda && (property == null || Boolean.parseBoolean(property.trim()));
    }
}
