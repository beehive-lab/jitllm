package org.beehive.jitllm.model.gptoss;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.beehive.jitllm.backend.cpu.GptOssForward;
import org.beehive.jitllm.backend.tornado.kernels.GptOssKernelsHostView;
import org.beehive.jitllm.tensor.standard.MXFP4FloatTensor;
import org.junit.Test;

/**
 * The pieces of gpt-oss arithmetic that are easy to get subtly wrong, against their definitions.
 */
public class GptOssArithmeticTest {

    private static GptOssConfiguration config(int layers) {
        return new GptOssConfiguration(
                "Q8_0", 2880, layers, 64, 8, 64, 201088, 131072, 1024, 32, 4, 2880, 1e-5f, 150000f,
                32f, 4096, 32f, 1f, 128, 2);
    }

    @Test
    public void evenLayersSlideOverTheLast128PositionsAndOddLayersSeeEverything() {
        GptOssConfiguration c = config(24);
        assertTrue(c.isSlidingWindowLayer(0));
        assertFalse(c.isSlidingWindowLayer(1));
        assertEquals(0, c.windowStart(0, 127));
        assertEquals(1, c.windowStart(0, 128));
        assertEquals(0, c.windowStart(1, 500));
        assertEquals(64 * 64, c.queryDim());
        assertEquals(512, c.kvDim());
    }

    @Test
    public void swigluClampsTheGateAboveAndTheLinearPartBothWays() {
        float alpha = 1.702f;
        float g = 9f; // capped at 7
        float u = -10f; // clamped to -7
        float expected = 7f / (1f + (float) Math.exp(-alpha * 7f)) * (-7f + 1f);
        assertEquals(expected, GptOssForward.swiglu(g, u), 1e-5f);
        float g2 = -3f; // not capped below
        float expected2 = -3f / (1f + (float) Math.exp(alpha * 3f)) * (0.5f + 1f);
        assertEquals(expected2, GptOssForward.swiglu(g2, 0.5f), 1e-6f);
    }

    @Test
    public void theDeviceDecodeOfANibbleAndAnExponentIsTheHostOne() {
        for (int n = 0; n < 16; n++) {
            assertEquals(MXFP4FloatTensor.KVALUES[n], GptOssKernelsHostView.fp4(n), 0f);
        }
        for (int e = 2; e < 255; e++) {
            assertEquals(MXFP4FloatTensor.scale(e), GptOssKernelsHostView.e8m0Half(e), 0f);
            assertEquals((float) Math.pow(2, e - 128), MXFP4FloatTensor.scale(e), 0f);
        }
    }
}
