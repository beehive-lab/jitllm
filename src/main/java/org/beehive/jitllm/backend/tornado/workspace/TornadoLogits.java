package org.beehive.jitllm.backend.tornado.workspace;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
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

            /** Reused across tokens: the workspace keeps one view per logits array. */
            private HostLogits host;

            @Override
            public Logits toHost() {
                int n = array.getSize();
                if (host == null || host.scores().length != n) {
                    host = new HostLogits(new float[n]);
                }
                MemorySegment.copy(
                        array.getSegment(), ValueLayout.JAVA_FLOAT, 0, host.scores(), 0, n);
                return host;
            }

            @Override
            public int argmax() {
                return toHost().argmax();
            }
        };
    }

    /** Device logits copied to the heap, with the device view's arithmetic. */
    private record HostLogits(float[] scores) implements Logits {

        @Override
        public int size() {
            return scores.length;
        }

        @Override
        public float get(int index) {
            return scores[index];
        }

        @Override
        public void divideInPlace(int from, int to, float value) {
            FloatArrayUtils.divideInPlace(scores, from, to, value);
        }

        @Override
        public void softmaxInPlace(int from, int to) {
            FloatArrayUtils.softmaxInPlace(scores, from, to);
        }

        /**
         * Over the array itself: through size() and get() a native image keeps two calls per score.
         */
        @Override
        public int argmax() {
            int maxIndex = 0;
            float maxValue = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < scores.length; i++) {
                if (scores[i] > maxValue) {
                    maxValue = scores[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }
    }
}
