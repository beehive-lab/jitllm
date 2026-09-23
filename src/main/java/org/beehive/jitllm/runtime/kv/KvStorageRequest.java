package org.beehive.jllm.runtime.kv;

/**
 * What a runtime needs of a key/value pool, stated without naming a backend.
 *
 * <p>Every field is a shape the manager already knows: it is the argument list the three call sites
 * were passing to a backend constructor directly, named as a value so that they no longer have to
 * name the constructor.
 *
 * @param totalBlocks blocks to allocate across every slot
 * @param blocksPerSlot table entries per slot — the stride the in-kernel walk multiplies by
 * @param maxSlots how many leases the table addresses
 * @param blockSizeTokens tokens one block holds; fixed for the store's lifetime
 * @param numberOfLayers layers the pool spans
 * @param kvDim key/value values per token per layer
 * @param fp16KeyValue allocate the half-precision pair instead of the FP32 pair. Never both: that
 *     would double the largest allocation in the process for nothing
 */
public record KvStorageRequest(
        int totalBlocks,
        int blocksPerSlot,
        int maxSlots,
        int blockSizeTokens,
        int numberOfLayers,
        int kvDim,
        boolean fp16KeyValue) {

    public KvStorageRequest {
        if (totalBlocks <= 0
                || blocksPerSlot <= 0
                || maxSlots <= 0
                || blockSizeTokens <= 0
                || numberOfLayers <= 0
                || kvDim <= 0) {
            throw new IllegalArgumentException(
                    "every key/value pool dimension must be positive,"
                            + " got totalBlocks="
                            + totalBlocks
                            + " blocksPerSlot="
                            + blocksPerSlot
                            + " maxSlots="
                            + maxSlots
                            + " blockSizeTokens="
                            + blockSizeTokens
                            + " numberOfLayers="
                            + numberOfLayers
                            + " kvDim="
                            + kvDim);
        }
    }
}
