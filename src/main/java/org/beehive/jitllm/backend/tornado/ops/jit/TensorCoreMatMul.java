package org.beehive.jitllm.backend.tornado.ops.jit;

import org.beehive.jitllm.backend.tornado.ops.DecodedMatMul;
import org.beehive.jitllm.backend.tornado.ops.Step;
import org.beehive.jitllm.backend.tornado.ops.Target;
import org.beehive.jitllm.runtime.backend.DeviceCapability;

/**
 * The tensor-core batched prefill on the JIT GEMM: JIT kernels decode the weights to FP16, then the
 * JIT tensor-core GEMM multiplies. Used on any GPU with tensor cores.
 */
public final class TensorCoreMatMul extends DecodedMatMul {

    @Override
    public String library() {
        return "none";
    }

    @Override
    public boolean accepts(Step step, Target target) {
        return target.supports(DeviceCapability.TENSOR_CORE_MMA);
    }

    @Override
    protected boolean nativeGemm() {
        return false;
    }
}
