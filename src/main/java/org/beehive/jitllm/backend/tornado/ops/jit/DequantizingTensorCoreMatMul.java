package org.beehive.jitllm.backend.tornado.ops.jit;

import org.beehive.jitllm.backend.tornado.ops.DequantizingMatMul;
import org.beehive.jitllm.backend.tornado.ops.Step;
import org.beehive.jitllm.backend.tornado.ops.Target;

/**
 * The tensor-core batched prefill over quantized weights decoded one projection at a time, the GEMM
 * the JIT one. Used on any GPU with tensor cores.
 */
public final class DequantizingTensorCoreMatMul extends DequantizingMatMul {

    @Override
    public String library() {
        return "none";
    }

    @Override
    public boolean accepts(Step step, Target target) {
        return tensorCoresOverDecoded(step, target);
    }

    @Override
    protected boolean nativeGemm() {
        return false;
    }
}
