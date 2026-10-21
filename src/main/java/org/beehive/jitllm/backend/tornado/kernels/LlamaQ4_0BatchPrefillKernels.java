package org.beehive.jitllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/** Kernels of the Llama Q4_0 batched prefill that runs its projections through cuBLAS. */
public final class LlamaQ4_0BatchPrefillKernels {

    private LlamaQ4_0BatchPrefillKernels() {}

    /**
     * {@code out[i] = silu(gate[i]) * up[i]} for the first {@code n} elements, written in FP16 for
     * the down projection. Gate and up are separate GEMM outputs: computing them into one packed
     * buffer would need an FP16 copy of both weights at once, twice the scratch.
     */
    public static void swiGLUFP16(
            KernelContext context, HalfFloatArray out, FloatArray gate, FloatArray up, int n) {
        int i = context.globalIdx;
        if (i < n) {
            float g = gate.get(i);
            float silu = g / (1.0f + TornadoMath.exp(-g));
            out.set(i, new HalfFloat(silu * up.get(i)));
        }
    }
}
