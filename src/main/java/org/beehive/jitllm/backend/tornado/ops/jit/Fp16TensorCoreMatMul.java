package org.beehive.jitllm.backend.tornado.ops.jit;

import org.beehive.jitllm.backend.tornado.ops.Fp16MatMul;
import org.beehive.jitllm.backend.tornado.ops.Step;
import org.beehive.jitllm.backend.tornado.ops.Target;

/**
 * The tensor-core batched prefill over F16 weights on the JIT GEMMs, which read the weights where
 * they are. Used on any GPU with tensor cores.
 */
public final class Fp16TensorCoreMatMul extends Fp16MatMul {

    @Override
    public String library() {
        return "none";
    }

    @Override
    public boolean accepts(Step step, Target target) {
        return tensorCoresOverF16(step, target);
    }

    @Override
    protected boolean nativeGemm() {
        return false;
    }
}
