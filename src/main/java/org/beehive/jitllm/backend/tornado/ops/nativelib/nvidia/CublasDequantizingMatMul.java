package org.beehive.jitllm.backend.tornado.ops.nativelib.nvidia;

import org.beehive.jitllm.backend.tornado.ops.DequantizingMatMul;
import org.beehive.jitllm.backend.tornado.ops.Step;
import org.beehive.jitllm.backend.tornado.ops.Target;

/**
 * The tensor-core batched prefill over quantized weights decoded one projection at a time, the GEMM
 * on cuBLAS. Used when native libraries are allowed, the GPU has tensor cores and cuBLAS loads.
 */
public final class CublasDequantizingMatMul extends DequantizingMatMul {

    @Override
    public String library() {
        return "cuBLAS";
    }

    @Override
    public boolean accepts(Step step, Target target) {
        return target.policy().nativeLibraries()
                && tensorCoresOverDecoded(step, target)
                && NativePrefillSupport.cublasAvailable();
    }

    @Override
    protected boolean nativeGemm() {
        return true;
    }
}
