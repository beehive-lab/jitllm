package org.beehive.jitllm.inference;

import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * The scores a sampler reads — indexed floats, and nothing else.
 *
 * <p>Before it, {@code Sampler.sampleToken(Object)} took an untyped argument and switched on {@code
 * instanceof FloatTensor} / {@code instanceof FloatArray} — which is how the samplers came to name
 * a TornadoVM type at all. The switch is gone with the parameter.
 */
public interface Logits {

    /** How many scores there are — the vocabulary size. */
    int size();

    /** The score at {@code index}. */
    float get(int index);

    /**
     * Divides {@code [from, to)} by {@code value} — temperature scaling.
     *
     * <p>An operation rather than a setter, and this is deliberate. The host and device paths have
     * <b>different implementations</b> of this and of the softmax below, and one is vectorised. A
     * single implementation over {@code get}/{@code set} would change the summation order and with
     * it the arithmetic, which this milestone is not entitled to do. Each adapter delegates to the
     * code that ran before.
     */
    void divideInPlace(int from, int to, float value);

    /** Softmaxes {@code [from, to)} in place. Same reasoning as {@link #divideInPlace}. */
    void softmaxInPlace(int from, int to);

    /**
     * These scores as a view a sampler can read element by element: a host view is one already and
     * returns itself; a device view copies its scores to the heap in one transfer and returns that
     * copy, so {@link #divideInPlace} and {@link #softmaxInPlace} on the result leave the device
     * array as it was.
     *
     * <p>Over a device array, {@link #get} is a {@code MemorySegment} access per element whose
     * checks only a JIT compiler removes; in a GraalVM native image every pass over the vocabulary
     * cost about a millisecond, and temperature sampling makes several per token.
     */
    default Logits toHost() {
        return this;
    }

    /** The index of the highest score, the first one on ties — greedy sampling. */
    default int argmax() {
        int maxIndex = 0;
        float maxValue = Float.NEGATIVE_INFINITY;
        for (int i = 0, n = size(); i < n; i++) {
            float value = get(i);
            if (value > maxValue) {
                maxValue = value;
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /** A view over host logits. */
    static Logits of(FloatTensor tensor) {
        return new Logits() {
            @Override
            public int size() {
                return tensor.size();
            }

            @Override
            public float get(int index) {
                return tensor.getFloat(index);
            }

            @Override
            public void divideInPlace(int from, int to, float value) {
                tensor.divideInPlace(from, to, value);
            }

            @Override
            public void softmaxInPlace(int from, int to) {
                tensor.softmaxInPlace(from, to);
            }
        };
    }
}
