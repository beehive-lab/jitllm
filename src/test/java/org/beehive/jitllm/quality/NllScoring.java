package org.beehive.jitllm.quality;

/**
 * Teacher-forced negative log-likelihood, and the alignment it depends on.
 *
 * <p>Separate from anything that loads a model so the arithmetic and the alignment can be tested on
 * their own, deterministically and in milliseconds. Everything here accumulates in {@code double}:
 * a 256-token passage over a 150k vocabulary is a lot of single-precision addition, and the two
 * runs being compared differ by less than that would cost.
 */
public final class NllScoring {

    private NllScoring() {}

    /**
     * {@code log sum exp} over a logits row, shifted by the maximum.
     *
     * <p>The shift is not optional: these logits reach tens in magnitude, and {@code exp} of that
     * overflows a float long before the sum is interesting.
     */
    public static double logSumExp(float[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (float value : logits) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException("non-finite logit: " + value);
            }
            max = Math.max(max, value);
        }
        double sum = 0.0;
        for (float value : logits) {
            sum += Math.exp(value - max);
        }
        return max + Math.log(sum);
    }

    /** {@code -log p(target)} from one row of logits. */
    public static double negativeLogLikelihood(float[] logits, int target) {
        if (target < 0 || target >= logits.length) {
            throw new IllegalArgumentException(
                    "target " + target + " outside the vocabulary of " + logits.length);
        }
        return logSumExp(logits) - logits[target];
    }

    /**
     * The positions a passage of {@code tokens} contributes, as {@code (position, target)} pairs.
     *
     * <p>The row produced <b>after</b> consuming {@code tokens[t]} predicts {@code tokens[t+1]}, so
     * a passage of {@code n} tokens scores {@code n - 1} of them and the first token is never a
     * target — it is the unscored prefix, and it is the whole of it. Nothing is padded, so nothing
     * is excluded.
     */
    public static int[][] scoredPositions(int[] tokens) {
        int[][] pairs = new int[Math.max(0, tokens.length - 1)][2];
        for (int t = 0; t + 1 < tokens.length; t++) {
            pairs[t][0] = t;
            pairs[t][1] = tokens[t + 1];
        }
        return pairs;
    }
}
