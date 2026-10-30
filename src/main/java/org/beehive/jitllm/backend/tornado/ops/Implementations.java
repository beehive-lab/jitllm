package org.beehive.jitllm.backend.tornado.ops;

import java.util.List;
import org.beehive.jitllm.backend.tornado.ops.jit.TensorCoreMatMul;
import org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia.CublasMatMul;

/**
 * Implementations that need nothing from the layer that uses them, best first. The first one that
 * accepts a step runs it, so vendor libraries come before the JIT kernels.
 */
public final class Implementations {

    private Implementations() {}

    private static final List<Implementation> TENSOR_CORE_PREFILL =
            List.of(new CublasMatMul(), new TensorCoreMatMul());

    /** The batched prefill on a tensor-core GPU: cuBLAS, then the JIT tensor-core GEMM. */
    public static List<Implementation> tensorCorePrefill() {
        return TENSOR_CORE_PREFILL;
    }
}
