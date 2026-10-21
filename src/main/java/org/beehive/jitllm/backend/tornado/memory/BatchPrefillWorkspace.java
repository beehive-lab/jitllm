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
 * @param decodedWeights the projection weights a resident copy would hold, 0 for a family that
 *     decodes none
 * @param residentBytes the total with every projection kept decoded
 * @param perChunkBytes the total with one per-chunk scratch instead
 */
public record BatchPrefillWorkspace(long decodedWeights, long residentBytes, long perChunkBytes) {

    public BatchPrefillWorkspace {
        if (decodedWeights < 0 || residentBytes < 0 || perChunkBytes < 0) {
            throw new IllegalArgumentException("workspace sizes cannot be negative");
        }
    }

    /** A family that decodes no projections: the same buffers either way. */
    public static BatchPrefillWorkspace fixed(long bytes) {
        return new BatchPrefillWorkspace(0, bytes, bytes);
    }

    /** Whether the family decodes projections, so residency is a choice at all. */
    public boolean decodesProjections() {
        return decodedWeights > 0;
    }

    /** The bytes for the given residency. */
    public long bytes(boolean resident) {
        return resident ? residentBytes : perChunkBytes;
    }
}
