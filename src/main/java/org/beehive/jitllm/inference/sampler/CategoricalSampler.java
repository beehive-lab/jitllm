package org.beehive.jllm.inference.sampler;

import java.util.random.RandomGenerator;
import org.beehive.jllm.inference.Logits;

/**
 * A sampler that samples from a categorical distribution. Supports both FloatTensor and FloatArray
 * implementations.
 */
public record CategoricalSampler(RandomGenerator rng) implements Sampler {

    /** Samples from a cumulative distribution. */
    @Override
    public int sampleToken(Logits logits) {
        // sample index from probabilities (they must sum to 1!)
        float random0to1 = rng.nextFloat(1f);
        float cdf = 0.0f;
        for (int i = 0; i < logits.size(); i++) {
            cdf += logits.get(i);
            if (random0to1 < cdf) {
                return i;
            }
        }
        return logits.size() - 1; // in case of rounding errors
    }
}
