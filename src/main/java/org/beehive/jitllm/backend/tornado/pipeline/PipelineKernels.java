package org.beehive.jitllm.backend.tornado.pipeline;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** Kernels for moving the hidden state between pipeline stages. */
public final class PipelineKernels {

    private PipelineKernels() {}

    /** {@code dst[i] = src[i]} for the first {@code n} elements. */
    public static void copy(KernelContext context, FloatArray src, FloatArray dst, int n) {
        int i = context.globalIdx;
        if (i < n) {
            dst.set(i, src.get(i));
        }
    }
}
