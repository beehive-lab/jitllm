package org.beehive.jitllm.runtime.backend;

/**
 * What a pool has and what is left of it.
 *
 * <p>They describe <b>the pool this runtime sized and what it has handed out</b> — not what the
 * device has free. A physical free-memory query was considered for this task and rejected:
 * TornadoVM recycles buffers, works within a configured device-memory budget, and releases at the
 * process level with a delay, so "bytes free" is not a number admission can be correct about. It
 * would also change admission behaviour, which is precisely what this task's acceptance forbids.
 * Physical-memory-aware admission is future work with its own measurements and its own contract —
 * <b>not</b> a widening of this record.
 *
 * @param totalBlocks blocks the pool was sized for
 * @param freeBlocks blocks not currently leased
 * @param blockSizeTokens tokens one block holds
 * @param bytesPerBlock device bytes one block occupies, across all layers
 */
public record KvCapacity(int totalBlocks, int freeBlocks, int blockSizeTokens, long bytesPerBlock) {

    public KvCapacity {
        if (totalBlocks < 0 || freeBlocks < 0 || freeBlocks > totalBlocks) {
            throw new IllegalArgumentException(
                    "free blocks must be within 0.." + totalBlocks + ", got " + freeBlocks);
        }
    }

    /** Blocks currently leased out. */
    public int usedBlocks() {
        return totalBlocks - freeBlocks;
    }

    /** Bytes the pool occupies in total, leased or not. */
    public long totalBytes() {
        return totalBlocks * bytesPerBlock;
    }

    /** Bytes backing currently leased blocks. */
    public long usedBytes() {
        return usedBlocks() * bytesPerBlock;
    }

    /** Tokens the pool can hold in total. */
    public long totalTokens() {
        return (long) totalBlocks * blockSizeTokens;
    }
}
