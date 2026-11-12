package org.beehive.jitllm.backend.tornado.ops;

import java.util.List;
import org.beehive.jitllm.backend.tornado.ops.jit.DequantizingTensorCoreMatMul;
import org.beehive.jitllm.backend.tornado.ops.jit.Fp16TensorCoreMatMul;
import org.beehive.jitllm.backend.tornado.ops.jit.TensorCoreMatMul;
import org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia.CublasDequantizingMatMul;
import org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia.CublasFp16MatMul;
import org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia.CublasMatMul;

/**
 * Implementations that need nothing from the layer that uses them, best first. The first one that
 * accepts a step runs it, so vendor libraries come before the JIT kernels.
 */
public final class Implementations {

    private Implementations() {}

    private static final List<Implementation> TENSOR_CORE_PREFILL =
            List.of(new CublasMatMul(), new TensorCoreMatMul());

    private static final List<Implementation> F16_TENSOR_CORE_PREFILL =
            List.of(new CublasFp16MatMul(), new Fp16TensorCoreMatMul());

    private static final List<Implementation> DEQUANTIZING_TENSOR_CORE_PREFILL =
            List.of(new CublasDequantizingMatMul(), new DequantizingTensorCoreMatMul());

    /** The batched prefill on a tensor-core GPU: cuBLAS, then the JIT tensor-core GEMM. */
    public static List<Implementation> tensorCorePrefill() {
        return TENSOR_CORE_PREFILL;
    }

    /**
     * The batched prefill of an F16 model on a tensor-core GPU: cuBLAS, then the JIT GEMMs that
     * read the weights in place.
     */
    public static List<Implementation> f16TensorCorePrefill() {
        return F16_TENSOR_CORE_PREFILL;
    }

    /**
     * The batched prefill on a tensor-core GPU over quantized weights decoded one projection at a
     * time into a scratch: cuBLAS, then the JIT GEMM. Example: Llama Q4_0, Devstral Q4_K_M.
     */
    public static List<Implementation> dequantizingTensorCorePrefill() {
        return DEQUANTIZING_TENSOR_CORE_PREFILL;
    }
}
