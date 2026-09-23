package org.beehive.jllm.runtime.kv;

/**
 * The device-side storage a {@link KvCacheManager} hands out through leases — seen from above,
 * where the backend is not nameable.
 *
 * <p><b>Why this interface exists at all.</b> {@code runtime.kv} must not name TornadoVM [Rule 1],
 * and the only implementation, {@code TornadoKvStore}, is nothing but TornadoVM arrays. This is the
 * seam between them. It carries no per-token operation on purpose: a lease resolves once to a view,
 * and the hot path is the in-kernel block-table walk, never a call back through here.
 */
public interface KvStorage extends AutoCloseable {

    /**
     * Publishes the manager's block table to the device mirror.
     *
     * <p>Called when a lease is acquired or released — never per token. The array's identity is
     * fixed for the pool's life [C1], so this updates contents in place; an implementation that
     * reallocated here would invalidate every captured CUDA graph.
     */
    void publishBlockTable(int[] blockTable);

    /** Tokens per block. Fixed for the store's lifetime. */
    int blockSizeTokens();

    /** Table entries per slot — the stride the in-kernel walk multiplies the slot by. */
    int blocksPerSlot();

    /** How many device bytes one block occupies across all layers. */
    long bytesPerBlock();

    @Override
    void close();
}
