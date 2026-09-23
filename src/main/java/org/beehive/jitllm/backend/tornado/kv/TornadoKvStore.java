package org.beehive.jitllm.backend.tornado.kv;

import org.beehive.jitllm.runtime.kv.KvStorage;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The device side of the KV cache: one key pool, one value pool, one block table, shared by every
 * session leasing from the manager this store is attached to.
 *
 * <p><b>What changed.</b> Until now each {@code State} allocated its own contiguous key and value
 * arrays, so N live sessions meant N copies of the cache. The manager's leases were real but the
 * storage behind them was not shared. Here the arrays are allocated once, sized for the pool, and a
 * lease is a slot in them.
 *
 * <p><b>The layout</b>, which {@code KvBlockAddress} walks and #129's paged decode already used:
 *
 * <pre>
 * pool[ physBlock * (numLayers * blockSize * kvDim)
 *     + layer     * (blockSize * kvDim)
 *     + (pos % blockSize) * kvDim
 *     + c ]
 * </pre>
 *
 * <p>Precision follows the model's KV setting: FP32 arrays unless {@code jitllm.kvcache.fp16}, in
 * which case the half-precision pair is allocated instead and the FP32 pair is left null. Both are
 * never allocated at once — that would double the largest allocation in the process for nothing.
 */
public final class TornadoKvStore implements KvStorage {

    private final FloatArray keyPool;
    private final FloatArray valuePool;
    private final HalfFloatArray keyPoolFP16;
    private final HalfFloatArray valuePoolFP16;

    /** The device mirror of the manager's block table. Identity fixed for the store's life. */
    private final IntArray blockTable;

    private final int blockSizeTokens;
    private final int blocksPerSlot;
    private final int scratchBlock;
    private final int numLayers;
    private final int kvDim;
    private final long bytesPerBlock;
    private boolean closed;

    /**
     * @param totalBlocks leasable physical blocks; one more is allocated as the scratch block
     * @param blocksPerSlot table entries per slot — must match the manager's pool exactly
     * @param maxSlots slots the table addresses — must match the manager's pool exactly
     * @param blockSizeTokens tokens per block
     * @param numLayers transformer layers, all of which live inside one block
     * @param kvDim key/value values per token
     * @param fp16 allocate the half-precision pair instead of the FP32 pair
     */
    public TornadoKvStore(
            int totalBlocks,
            int blocksPerSlot,
            int maxSlots,
            int blockSizeTokens,
            int numLayers,
            int kvDim,
            boolean fp16) {
        // One block beyond the leasable range is the scratch block: inactive slots still run the
        // KV kernels every step, and their writes have to land somewhere no live sequence reads
        // (#129's arrangement). It is allocated here and never leased.
        this.scratchBlock = totalBlocks;
        long elements = (long) (totalBlocks + 1) * numLayers * blockSizeTokens * kvDim;
        if (elements > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "KV pool of "
                            + elements
                            + " elements exceeds what a single TornadoVM array can address; reduce"
                            + " maxConcurrentSessions or the context length");
        }
        this.blockSizeTokens = blockSizeTokens;
        this.blocksPerSlot = blocksPerSlot;
        this.numLayers = numLayers;
        this.kvDim = kvDim;
        this.bytesPerBlock = 2L * kvDim * blockSizeTokens * numLayers * (fp16 ? 2L : 4L);

        int size = (int) elements;
        if (fp16) {
            this.keyPool = null;
            this.valuePool = null;
            this.keyPoolFP16 = new HalfFloatArray(size);
            this.valuePoolFP16 = new HalfFloatArray(size);
            this.keyPoolFP16.init(new HalfFloat(0.f));
            this.valuePoolFP16.init(new HalfFloat(0.f));
        } else {
            this.keyPool = new FloatArray(size);
            this.valuePool = new FloatArray(size);
            this.keyPool.init(0.f);
            this.valuePool.init(0.f);
            this.keyPoolFP16 = null;
            this.valuePoolFP16 = null;
        }
        this.blockTable = new IntArray(maxSlots * blocksPerSlot);
    }

    /** FP32 key pool, or {@code null} when this store is half-precision. */
    public FloatArray keyPool() {
        return keyPool;
    }

    /** FP32 value pool, or {@code null} when this store is half-precision. */
    public FloatArray valuePool() {
        return valuePool;
    }

    /** FP16 key pool, or {@code null} when this store is single-precision. */
    public HalfFloatArray keyPoolFP16() {
        return keyPoolFP16;
    }

    /** FP16 value pool, or {@code null} when this store is single-precision. */
    public HalfFloatArray valuePoolFP16() {
        return valuePoolFP16;
    }

    /** The device block table every paged KV kernel reads. */
    public IntArray blockTable() {
        return blockTable;
    }

    /** {@code blockSize | (blocksPerSlot << 16)}, packed as #129 packs it to fit the arg limit. */
    public int blockCfg() {
        return blockSizeTokens | (blocksPerSlot << 16);
    }

    /** {@code numLayers * blockSize * kvDim} — the distance between two blocks. */
    public int blockStride() {
        return numLayers * blockSizeTokens * kvDim;
    }

    /**
     * {@inheritDoc}
     *
     * <p><b>Unmapped entries are translated to the scratch block here</b>, and only here. The host
     * table keeps {@link org.beehive.jitllm.runtime.kv.BlockPool#UNMAPPED} because that is the
     * accounting truth — "no lease holds this slot" — while the device needs an index it can safely
     * write to, since an inactive slot still executes the KV kernels every step. Doing the
     * translation on the host instead would collapse the two meanings and lose the distinction the
     * pool's own tests assert.
     */
    @Override
    public void publishBlockTable(int[] table) {
        if (closed) {
            throw new IllegalStateException("this KV store is closed");
        }
        if (table.length != blockTable.getSize()) {
            throw new IllegalArgumentException(
                    "block table shape changed: manager has "
                            + table.length
                            + " entries, the device mirror has "
                            + blockTable.getSize()
                            + ". The two are sized together at construction and cannot diverge");
        }
        for (int i = 0; i < table.length; i++) {
            int entry = table[i];
            blockTable.set(
                    i,
                    entry == org.beehive.jitllm.runtime.kv.BlockPool.UNMAPPED ? scratchBlock : entry);
        }
    }

    /** The block index unmapped table entries are published as. Never leased, never read. */
    public int scratchBlock() {
        return scratchBlock;
    }

    @Override
    public int blockSizeTokens() {
        return blockSizeTokens;
    }

    @Override
    public int blocksPerSlot() {
        return blocksPerSlot;
    }

    @Override
    public long bytesPerBlock() {
        return bytesPerBlock;
    }

    /**
     * Marks the store unusable. The arrays themselves are released by the collector once the plans
     * that bound them are gone — TornadoVM device memory follows the plan, not this object.
     */
    @Override
    public void close() {
        closed = true;
    }
}
