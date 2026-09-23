package org.beehive.jllm.program;

/**
 * What a compiled program's device arrays were sized from.
 *
 * @param contextLength the longest sequence the key/value storage was sized for
 * @param numberOfLayers how many transformer layers the program describes
 * @param kvBlockSize elements per key/value block
 * @param kvBlockCount how many blocks the pool holds
 * @param maxPrefillBatch the largest prefill batch the staging buffers were sized for
 * @param maxBatch the largest number of concurrent sequences, {@code B}
 */
public record CapacityShape(
        int contextLength,
        int numberOfLayers,
        int kvBlockSize,
        int kvBlockCount,
        int maxPrefillBatch,
        int maxBatch) {

    public CapacityShape {
        requirePositive("contextLength", contextLength);
        requirePositive("numberOfLayers", numberOfLayers);
        requirePositive("kvBlockSize", kvBlockSize);
        requirePositive("kvBlockCount", kvBlockCount);
        requirePositive("maxPrefillBatch", maxPrefillBatch);
        requirePositive("maxBatch", maxBatch);
    }

    private static void requirePositive(String what, int value) {
        if (value <= 0) {
            throw new IllegalArgumentException(what + " must be positive: " + value);
        }
    }
}
