package org.beehive.jllm.inference.op;

/**
 * The head geometry one attention call works over.
 *
 * <p>Nine numbers rather than "head size", because the families genuinely disagree about what a
 * head size is. Llama has one: queries, keys and values are all {@code dim / heads}. Qwen3 has
 * three — {@code attention.key_length}, {@code attention.value_length} and the stride its query
 * buffer is laid out with — and they need not be equal to each other or to {@code dim / heads}.
 * Collapsing them would work on Llama and quietly mis-address Qwen3.
 *
 * <p>Bundled into a value so the call site reads as geometry rather than as nine loose ints, and so
 * a family's shape can be built once and asserted against.
 *
 * @param heads number of query heads
 * @param kvMul query heads per key/value head; 1 for multi-head, more for grouped query
 * @param queryHeadStride distance between one head's queries and the next in the query buffer
 * @param keyHeadStride distance between one key/value head and the next inside a cached row
 * @param keyDotLength how many components of a key a score is the dot product over
 * @param valueHeadDim width of one head's value, and the stride of the output buffer
 * @param kvDim width of one cached row: all key/value heads together
 * @param contextLength stride between heads in the score buffer
 * @param scoreScaling whether a raw score is divided or multiplied by {@link #scoreScale}
 * @param scoreScale the divisor or the multiplier, depending on {@link #scoreScaling}
 * @param window how many positions back a query may attend, or 0 for full causal attention — the
 *     sliding-window layers Gemma4 alternates with full ones
 */
public record AttentionShape(
        int heads,
        int kvMul,
        int queryHeadStride,
        int keyHeadStride,
        int keyDotLength,
        int valueHeadDim,
        int kvDim,
        int contextLength,
        ScoreScaling scoreScaling,
        float scoreScale,
        int window) {

    /**
     * How a raw score becomes a scaled score.
     *
     * <p>Two cases rather than one, because Granite genuinely replaces the division: it applies a
     * µP {@code attentionScale} <b>instead of</b> {@code 1/sqrt(headDim)}, not in addition to it.
     * Expressing Granite as a division by the reciprocal would round, and the outputs would stop
     * being bit-identical to what it computes today. One branch on a final field, hoisted out of
     * the score loop once the record is inlined, executes exactly the one arithmetic operation each
     * family performed before.
     */
    public enum ScoreScaling {
        /** {@code score / scale} — the conventional {@code 1/sqrt(headDim)}. */
        DIVIDE,
        /** {@code score * scale} — Granite's µP attention multiplier. */
        MULTIPLY
    }

    public AttentionShape {
        if (heads <= 0 || kvMul <= 0) {
            throw new IllegalArgumentException(
                    "heads and kvMul must be positive: " + heads + ", " + kvMul);
        }
        if (queryHeadStride <= 0 || keyHeadStride <= 0 || keyDotLength <= 0 || valueHeadDim <= 0) {
            throw new IllegalArgumentException("head dimensions must be positive");
        }
        if (kvDim <= 0 || contextLength <= 0) {
            throw new IllegalArgumentException("kvDim and contextLength must be positive");
        }
        java.util.Objects.requireNonNull(scoreScaling, "scoreScaling");
        if (window < 0) {
            throw new IllegalArgumentException(
                    "window must not be negative; full attention is 0: " + window);
        }
    }

    /**
     * The first position a query at {@code position} may attend to.
     *
     * <p>Zero for full causal attention, which is what every family but Gemma4 passes, and what the
     * pre-window implementations computed literally.
     */
    public int windowStart(int position) {
        return window == 0 ? 0 : Math.max(0, position - window + 1);
    }

    /**
     * Applies this shape's score scaling.
     *
     * <p>Exactly one division or one multiplication, the same one the family performed before it
     * was named.
     */
    public float scaleScore(float rawScore) {
        return scoreScaling == ScoreScaling.DIVIDE ? rawScore / scoreScale : rawScore * scoreScale;
    }

    /**
     * The uniform case: queries, keys and values all {@code headSize} wide.
     *
     * <p>Llama, Mistral, Devstral, Qwen2, Granite and Phi3.
     */
    public static AttentionShape uniform(
            int heads, int kvMul, int headSize, int kvDim, int contextLength, float scoreDivisor) {
        return new AttentionShape(
                heads,
                kvMul,
                headSize,
                headSize,
                headSize,
                headSize,
                kvDim,
                contextLength,
                ScoreScaling.DIVIDE,
                scoreDivisor,
                0);
    }

    /** The uniform case with Granite's µP score multiplier in place of the conventional divisor. */
    public static AttentionShape uniformScaled(
            int heads,
            int kvMul,
            int headSize,
            int kvDim,
            int contextLength,
            float scoreMultiplier) {
        return new AttentionShape(
                heads,
                kvMul,
                headSize,
                headSize,
                headSize,
                headSize,
                kvDim,
                contextLength,
                ScoreScaling.MULTIPLY,
                scoreMultiplier,
                0);
    }
}
