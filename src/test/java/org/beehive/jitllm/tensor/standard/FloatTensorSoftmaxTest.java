package org.beehive.jitllm.tensor.standard;

import static org.junit.Assert.assertEquals;

import java.util.Random;
import org.junit.Test;

/** The softmax that skips the exponentials that underflow, against the one that takes them all. */
public class FloatTensorSoftmaxTest {

    @Test
    public void theExponentialAtTheThresholdIsZero() {
        assertEquals(0, Float.floatToRawIntBits((float) Math.exp(FloatTensor.EXP_UNDERFLOW)));
        assertEquals(
                0, Float.floatToRawIntBits((float) Math.exp((double) FloatTensor.EXP_UNDERFLOW)));
    }

    @Test
    public void skippingTheUnderflowsChangesNoBit() {
        int n = 154880;
        Random rng = new Random(7);
        float[] values = new float[n];
        for (int i = 0; i < n; i++) {
            // Temperature-scaled logits: a wide spread, with values on and around the threshold.
            values[i] = (float) rng.nextGaussian() * 40f;
        }
        values[1] = 200f;
        values[2] = 200f + FloatTensor.EXP_UNDERFLOW;
        values[3] = Math.nextUp(values[2]);
        values[4] = Math.nextDown(values[2]);

        FloatTensor skipping = new ArrayFloatTensor(values.clone());
        skipping.softmaxInPlace(0, n);

        FloatTensor all = new ArrayFloatTensor(values.clone());
        float max = all.max(0, n);
        all.mapInPlace(0, n, f -> (float) Math.exp(f - max));
        all.divideInPlace(0, n, all.sum(0, n));

        for (int i = 0; i < n; i++) {
            assertEquals(
                    "at " + i,
                    Float.floatToRawIntBits(all.getFloat(i)),
                    Float.floatToRawIntBits(skipping.getFloat(i)));
        }
    }
}
