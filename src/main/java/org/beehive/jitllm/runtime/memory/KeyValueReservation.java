package org.beehive.jitllm.runtime.memory;

import org.beehive.jitllm.api.Experimental;

/**
 * How a model's key/value storage is provisioned, which a memory plan must know to size it.
 *
 * <p>Two shapes, and they cost differently:
 *
 * <ul>
 *   <li><b>pooled</b> — one device pool shared by every session, reserved when the model loads for
 *       {@code sessions} sequences of the full context, plus one scratch block that inactive slots
 *       write into;
 *   <li><b>private</b> — each session allocates its own cache when it is opened. The plan charges
 *       one session's; each further open session adds its own.
 * </ul>
 *
 * <p>Both are block-rounded: storage is laid out in blocks of {@code blockSizeTokens}, so a context
 * that is not a multiple of the block size occupies the next whole block.
 *
 * @param sessions how many sessions may be open at once, at least 1
 * @param pooled whether the sessions share one pool reserved at load
 * @param blockSizeTokens tokens per storage block, at least 1
 * @param fp16 whether entries are stored in half precision rather than single
 */
@Experimental
public record KeyValueReservation(int sessions, boolean pooled, int blockSizeTokens, boolean fp16) {

    /** The block size every key/value store in the tree is laid out with. */
    public static final int BLOCK_SIZE_TOKENS = 16;

    public KeyValueReservation {
        if (sessions < 1 || blockSizeTokens < 1) {
            throw new IllegalArgumentException(
                    "a key/value reservation needs at least one session and a positive block size");
        }
    }

    /** One session with its own cache — the shape of a plan made without knowing the pool. */
    public static KeyValueReservation singlePrivate(boolean fp16) {
        return new KeyValueReservation(1, false, BLOCK_SIZE_TOKENS, fp16);
    }

    /** Bytes one stored key or value element occupies. */
    public int bytesPerElement() {
        return fp16 ? 2 : 4;
    }

    /**
     * Elements in <b>one</b> of the key and value arrays this reservation allocates up front.
     *
     * @param contextLength tokens one sequence may reach
     * @param keyValueLayers layers that hold key/value entries
     * @param kvDim key/value values per token per layer
     */
    public long elementsPerArray(int contextLength, int keyValueLayers, int kvDim) {
        long blocksPerSequence = (contextLength + (long) blockSizeTokens - 1) / blockSizeTokens;
        long blocks = pooled ? sessions * blocksPerSequence + 1 : blocksPerSequence;
        return blocks * blockSizeTokens * keyValueLayers * kvDim;
    }
}
