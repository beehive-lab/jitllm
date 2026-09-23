package org.beehive.jitllm.backend.tornado.workspace;

import org.beehive.jitllm.backend.tornado.utils.FloatArrayUtils;
import org.beehive.jitllm.inference.Logits;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** A {@link Logits} view over a device array. */
public final class TornadoLogits {

    private TornadoLogits() {}

    public static Logits of(FloatArray array) {
        return new Logits() {
            @Override
            public int size() {
                return array.getSize();
            }

            @Override
            public float get(int index) {
                return array.get(index);
            }

            @Override
            public void divideInPlace(int from, int to, float value) {
                FloatArrayUtils.divideInPlace(array, from, to, value);
            }

            @Override
            public void softmaxInPlace(int from, int to) {
                FloatArrayUtils.softmaxInPlace(array, from, to);
            }
        };
    }
}
