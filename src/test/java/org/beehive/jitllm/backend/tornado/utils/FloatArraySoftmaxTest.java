package org.beehive.jitllm.backend.tornado.utils;

import static org.junit.Assert.assertEquals;

import java.util.Random;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** The device logits' softmax, skipping the exponentials that underflow, against all of them. */
public class FloatArraySoftmaxTest {

    @Test
    public void skippingTheUnderflowsChangesNoBit() {
        int n = 154880;
        Random rng = new Random(7);
        FloatArray skipping = new FloatArray(n);
        float[] values = new float[n];
        for (int i = 0; i < n; i++) {
            values[i] = (float) rng.nextGaussian() * 40f;
        }
        values[1] = 200f;
        values[2] = 200f + FloatArrayUtils.EXP_UNDERFLOW;
        values[3] = Math.nextUp(values[2]);
        values[4] = Math.nextDown(values[2]);
        for (int i = 0; i < n; i++) {
            skipping.set(i, values[i]);
        }
        FloatArrayUtils.softmaxInPlace(skipping, 0, n);

        float max = Float.NEGATIVE_INFINITY;
        for (float v : values) {
            max = Math.max(max, v);
        }
        float[] all = new float[n];
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            all[i] = (float) Math.exp(values[i] - max);
            sum += all[i];
        }
        for (int i = 0; i < n; i++) {
            assertEquals(
                    "at " + i,
                    Float.floatToRawIntBits(all[i] / sum),
                    Float.floatToRawIntBits(skipping.get(i)));
        }
    }
}
