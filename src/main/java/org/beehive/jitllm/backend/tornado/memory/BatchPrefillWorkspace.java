package org.beehive.jitllm.backend.tornado.memory;

/**
 * What one session's batched prefill layers allocate on the device besides the staging every family
 * has: tensor-core attention operands, split-K partial products and, for a family that decodes its
 * weights to FP16 for the GEMMs, either a resident copy of every projection or one scratch decoded
 * again for each chunk.
 *
 * <p>Sized by the layer class that allocates it, from the same arithmetic its constructor uses, so
 * the prediction and the plan cannot drift apart.
 *
 * @param residentCopiesSize how many weights a resident copy would hold, 0 for a family with no
 *     choice between kept and refilled copies
 * @param residentBytes the total with every projection kept decoded
 * @param perChunkBytes the total with one per-chunk scratch instead
 */
public record BatchPrefillWorkspace(
        long residentCopiesSize, long residentBytes, long perChunkBytes) {

    public BatchPrefillWorkspace {
        if (residentCopiesSize < 0 || residentBytes < 0 || perChunkBytes < 0) {
            throw new IllegalArgumentException("workspace sizes cannot be negative");
        }
    }

    /** A family with no choice between kept and refilled copies: the same buffers either way. */
    public static BatchPrefillWorkspace withoutResidencyChoice(long bytes) {
        return new BatchPrefillWorkspace(0, bytes, bytes);
    }

    /** The sum of this workspace and {@code other}. */
    public BatchPrefillWorkspace plus(BatchPrefillWorkspace other) {
        return new BatchPrefillWorkspace(
                residentCopiesSize + other.residentCopiesSize,
                residentBytes + other.residentBytes,
                perChunkBytes + other.perChunkBytes);
    }

    /** This workspace plus {@code bytes} that are the same whether decoded weights stay or not. */
    public BatchPrefillWorkspace plus(long bytes) {
        return plus(withoutResidencyChoice(bytes));
    }

    /** Whether the family can either keep its projection copies or refill them per chunk. */
    public boolean hasResidencyChoice() {
        return residentCopiesSize > 0;
    }

    /** The bytes for the given residency. */
    public long bytes(boolean resident) {
        return resident ? residentBytes : perChunkBytes;
    }
}
