package org.beehive.jitllm.program.op;

import java.util.List;
import java.util.Objects;
import java.util.OptionalInt;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Scaled dot-product attention of one query set against a set of keys and values.
 *
 * <p><b>No cache is named here, and that is the point.</b> The keys and values are operands like
 * any other. Where they come from — a retained per-sequence cache addressed through a block table,
 * or an encoder's freshly computed projections — is the program's business, not the operation's.
 * Rule 14 forbids core abstractions from <i>requiring</i> a KV cache, and an attention operation
 * that named one would close the door on encoder-only and embedding models before they were asked
 * for.
 *
 * <p>Grouped query attention is expressed by {@code keyValueHeads &lt; heads}, not by a second
 * operation; multi-head attention is the case where they are equal.
 *
 * <p><b>Sliding-window attention is a parameter, not a second operation.</b> The mathematics is
 * identical; only the lower bound of the score loop moves, and a full-attention family passes the
 * empty window and executes what it always executed.
 *
 * @param query the query projection
 * @param keys the keys attended over
 * @param values the values attended over
 * @param output the attention result
 * @param heads the number of query heads
 * @param keyValueHeads the number of key/value heads; equal to {@code heads} for MHA, fewer for GQA
 * @param headDimension the size of one head
 * @param scale the value applied to the scores, in the direction {@code scoreScaling} names. Under
 *     {@link ScoreScaling#DIVIDE}, the default, it is the <b>divisor</b> — conventionally {@code
 *     sqrt(headDimension)}, which is what every Llama-shaped caller passes and what the validators
 *     check
 * @param scoreScaling whether {@code scale} divides the scores or multiplies them. Granite's µP
 *     multiplier <i>replaces</i> the conventional division rather than following it, and expressing
 *     it as a division by its reciprocal rounds — the mode is why {@code AttentionShape} already
 *     carries one on the host side
 * @param dataType the representation the keys and values are stored in; the FP16 key/value cache is
 *     this parameter, not a separate operation
 */
public record Attention(
        OperandRef query,
        OperandRef keys,
        OperandRef values,
        OperandRef output,
        int heads,
        int keyValueHeads,
        int headDimension,
        float scale,
        ScoreScaling scoreScaling,
        OptionalInt window,
        DataType dataType)
        implements Operation {

    /** How {@code scale} is applied to the attention scores. */
    public enum ScoreScaling {
        /** Scores are divided by {@code scale}; the conventional {@code 1/sqrt(headDim)} form. */
        DIVIDE,
        /** Scores are multiplied by {@code scale}; Granite's µP factor. */
        MULTIPLY
    }

    /** The conventional form: scores divided by {@code scale}. */
    public Attention(
            OperandRef query,
            OperandRef keys,
            OperandRef values,
            OperandRef output,
            int heads,
            int keyValueHeads,
            int headDimension,
            float scale,
            OptionalInt window,
            DataType dataType) {
        this(
                query,
                keys,
                values,
                output,
                heads,
                keyValueHeads,
                headDimension,
                scale,
                ScoreScaling.DIVIDE,
                window,
                dataType);
    }

    public Attention {
        Objects.requireNonNull(scoreScaling, "scoreScaling");
        Objects.requireNonNull(query, "query");
        Objects.requireNonNull(keys, "keys");
        Objects.requireNonNull(values, "values");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(window, "window");
        Objects.requireNonNull(dataType, "dataType");
        if (window.isPresent() && window.getAsInt() <= 0) {
            throw new IllegalArgumentException(
                    "a sliding window must be positive; full attention is"
                            + " the empty window, not zero: "
                            + window.getAsInt());
        }
        if (heads <= 0 || keyValueHeads <= 0 || headDimension <= 0) {
            throw new IllegalArgumentException(
                    "heads, keyValueHeads and headDimension must be"
                            + " positive: "
                            + heads
                            + ", "
                            + keyValueHeads
                            + ", "
                            + headDimension);
        }
        if (heads % keyValueHeads != 0) {
            throw new IllegalArgumentException(
                    "heads must be a multiple of keyValueHeads: " + heads + " % " + keyValueHeads);
        }
    }

    @Override
    public OperationKind kind() {
        return OperationKind.ATTENTION;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(query, keys, values);
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(output);
    }
}
