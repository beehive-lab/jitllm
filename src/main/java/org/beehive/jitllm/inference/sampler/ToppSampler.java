package org.beehive.jllm.inference.sampler;

import java.util.Comparator;
import java.util.random.RandomGenerator;
import org.beehive.jllm.inference.Logits;

/**
 * Top-p sampling (nucleus sampling) implementation supporting both FloatTensor and FloatArray.
 * Samples from the smallest set of tokens that exceed probability topp.
 */
public final class ToppSampler implements Sampler {

    final int[] indices;
    final float topp;
    final RandomGenerator rng;

    public ToppSampler(int maxNumberOfElements, float topp, RandomGenerator rng) {
        this.indices = new int[maxNumberOfElements];
        this.topp = topp;
        this.rng = rng;
    }

    static void swap(int[] array, int from, int to) {
        int tmp = array[from];
        array[from] = array[to];
        array[to] = tmp;
    }

    static void siftDown(int[] array, int from, int n, Comparator<Integer> comparator) {
        int prev = from, next;
        while ((next = 2 * prev + 1) < n) {
            int r = 2 * prev + 2;
            if (r < n && comparator.compare(array[r], array[next]) < 0) {
                next = r;
            }
            if (comparator.compare(array[next], array[prev]) < 0) {
                swap(array, prev, next);
                prev = next;
            } else {
                break;
            }
        }
    }

    /** Top-p sampling. */
    @Override
    public int sampleToken(Logits logits) {
        Comparator<Integer> comparator = (a, b) -> Float.compare(logits.get(b), logits.get(a));

        int n = logits.size();
        int head = 0;
        int tail = n - 1;
        // values smaller than (1 - topp) / (n - 1) cannot be part of the result
        // so for efficiency we crop these out as candidates before sorting
        float cutoff = (1.0f - topp) / (n - 1);
        for (int i = 0; i < indices.length; i++) {
            if (logits.get(i) >= cutoff) {
                indices[head++] = i;
            } else {
                indices[tail--] = i;
            }
        }

        return processTopP(logits, comparator, head);
    }

    /**
     * Common implementation for processing top-p sampling once indices are prepared. Uses a
     * type-specific value getter function to access tensor values.
     */
    private int processTopP(Logits logits, Comparator<Integer> comparator, int n0) {
        // build heap O(n0)
        for (int i = n0 / 2 - 1; i >= 0; --i) {
            siftDown(indices, i, n0, comparator);
        }

        // truncate the list where cumulative probability of the largest k elements exceeds topp
        // O(k lg n0)
        float cumulativeProb = 0.0f;
        int lastIndex = 0;
        for (int i = n0 - 1; i >= 0; i--) {
            swap(indices, 0, i);

            cumulativeProb += logits.get(indices[i]);
            if (cumulativeProb > topp) {
                lastIndex = i;
                break; // we've exceeded topp by including lastIndex
            }
            siftDown(indices, 0, i - 1, comparator);
        }

        // sample from the truncated list
        float r = rng.nextFloat(1f) * cumulativeProb;
        float cdf = 0.0f;
        for (int i = n0 - 1; i >= lastIndex; i--) {
            cdf += logits.get(indices[i]);
            if (r < cdf) {
                return indices[i];
            }
        }

        return indices[lastIndex]; // in case of rounding errors
    }
}
