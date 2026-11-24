package org.beehive.jitllm.backend.tornado.kernels;

/** The package-private decode helpers of {@link GptOssKernels}, for host tests. */
public final class GptOssKernelsHostView {

    private GptOssKernelsHostView() {}

    public static float fp4(int nibble) {
        return GptOssKernels.fp4(nibble);
    }

    public static float e8m0Half(int exponent) {
        return GptOssKernels.e8m0Half(exponent);
    }
}
