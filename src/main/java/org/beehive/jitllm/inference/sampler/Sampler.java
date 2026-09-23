package org.beehive.jllm.inference.sampler;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import org.beehive.jllm.Options;
import org.beehive.jllm.inference.Logits;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.tensor.standard.FloatTensor;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Generic interface for sampling tokens from probability distributions. Supports both FloatTensor
 * and FloatArray tensor implementations.
 */
@FunctionalInterface
public interface Sampler {

    /** Argmax implementation for FloatTensor. */
    Sampler TENSOR_ARGMAX =
            tensor -> {
                int maxIndex = 0;
                float maxValue = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < tensor.size(); i++) {
                    float value = tensor.get(i);
                    if (value > maxValue) {
                        maxValue = value;
                        maxIndex = i;
                    }
                }
                return maxIndex;
            };

    /**
     * Legacy ARGMAX for backward compatibility.
     *
     * @deprecated Use TENSOR_ARGMAX instead
     */
    @Deprecated Sampler ARGMAX = TENSOR_ARGMAX;

    /**
     * Creates and configures a sampler for token generation based on specified parameters.
     *
     * <p>This method selects an appropriate sampling strategy for next-token prediction in language
     * model inference. It supports several sampling approaches:
     *
     * <ul>
     *   <li>Greedy sampling (temperature = 0): Always selects the most probable token
     *   <li>Temperature sampling: Adjusts probability distribution sharpness
     *   <li>Top-p (nucleus) sampling: Considers only tokens comprising the top p probability mass
     * </ul>
     *
     * <p>The method handles both {@link FloatTensor} and {@link FloatArray} logits types to support
     * both CPU and GPU execution paths.
     *
     * @param vocabularySize The size of the model's vocabulary
     * @param temperature A value controlling randomness in sampling:
     *     <ul>
     *       <li>0.0f: No randomness (greedy sampling)
     *       <li>1.0f: Standard sampling from unmodified distribution
     *       <li>&lt;1.0f: More deterministic (sharper distribution)
     *       <li>&gt;1.0f: More random (flatter distribution)
     *     </ul>
     *
     * @param topp The cumulative probability threshold for nucleus sampling (0.0-1.0).
     *     <ul>
     *       <li>Values ≤0 or ≥1: Disables top-p sampling
     *       <li>Values in (0,1): Restricts sampling to tokens comprising the top p probability mass
     *     </ul>
     *
     * @param rngSeed Seed value for the random number generator to ensure reproducibility
     * @return A configured {@link Sampler} that implements the selected sampling strategy and
     *     handles both tensor and array-based logits
     * @throws IllegalArgumentException if logits are of an unsupported type
     */
    static Sampler selectSampler(int vocabularySize, float temperature, float topp, long rngSeed) {
        Sampler sampler;
        if (temperature == 0.0f) {
            // greedy argmax sampling: take the token with the highest probability
            sampler = Sampler.TENSOR_ARGMAX; // Use TENSOR_ARGMAX instead of ARGMAX
        } else {
            // we sample from this distribution to get the next token
            RandomGenerator rng = RandomGeneratorFactory.getDefault().create(rngSeed);
            Sampler innerSampler;
            // Determine whether to use top-p (nucleus) sampling
            if (topp <= 0 || topp >= 1) {
                // If topp is outside (0,1), use standard categorical sampling
                // This samples directly from the probability distribution
                innerSampler = new CategoricalSampler(rng);
            } else {
                // Use top-p (nucleus) sampling with the specified threshold
                // This restricts sampling to only the most likely tokens that
                // cumulatively comprise the top p probability mass
                innerSampler = new ToppSampler(vocabularySize, topp, rng);
            }

            // Create a sampler that:
            // 1. Applies temperature scaling to the logits
            // 2. Converts logits to probabilities using softmax
            // 3. Delegates the actual sampling to the appropriate inner sampler
            sampler =
                    logits -> {
                        // Temperature scaling, then softmax. Which implementation runs is the
                        // view's business: the host and device paths keep the code they had, so the
                        // arithmetic and its ordering are unchanged.
                        logits.divideInPlace(0, logits.size(), temperature);
                        logits.softmaxInPlace(0, logits.size());
                        return innerSampler.sampleToken(logits);
                    };
        }
        return sampler;
    }

    static Sampler createSampler(Model model, Options options) {
        float temperature =
                Float.isNaN(options.temperature())
                        ? (float) model.chatFormat().defaultTemperature()
                        : options.temperature();
        float topp =
                Float.isNaN(options.topp())
                        ? (float) model.chatFormat().defaultTopP()
                        : options.topp();
        return selectSampler(
                model.configuration().vocabularySize(), temperature, topp, options.seed());
    }

    /**
     * Find the index of the maximum value in a FloatArray.
     *
     * @param array The FloatArray to find the maximum value in
     * @return The index of the maximum value
     */

    /**
     * Sample a token from the provided tensor.
     *
     * @param tensor The tensor containing probabilities/logits
     * @return The selected token index
     */
    int sampleToken(Logits logits);
}
