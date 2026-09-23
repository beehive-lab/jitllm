package org.beehive.jitllm.runtime.kv;

import java.util.Arrays;
import java.util.BitSet;
import org.beehive.jitllm.runtime.backend.KvCapacity;
import org.beehive.jitllm.runtime.diagnostics.DiagnosticCode;

/**
 * The persistent block store behind the KV cache, and the one block table the kernels walk.
 *
 * <p><b>One array, allocated once</b> [C1]. Host code updates the table's <i>contents</i>; it never
 * replaces the array. CUDA-graph capture records buffer addresses, so a captured graph replays
 * against the addresses it captured — reallocating the table on a lease would silently invalidate
 * every captured graph. That is why {@link #blockTable()} hands out the same array for the pool's
 * whole life and why growing capacity is not something a lease can trigger.
 *
 * <p><b>The table is slot-major</b>, which is what lets a KV index say <i>which lease</i>:
 *
 * <pre>
 * blockTable[slot * blocksPerSlot + logicalBlock] = physical block, or UNMAPPED
 * </pre>
 *
 * <p><b>Physical blocks and logical slots are separate scarce resources.</b> A slot is an address
 * range in the table; a block is storage. Two leases never share a slot, and never share a block
 * either until prefix sharing arrives — at which point the same physical block will legitimately
 * appear under two slots, which this layout already expresses.
 *
 * <p>Not thread-safe on its own; {@link KvCacheManager} is the synchronizing owner.
 */
public final class BlockPool {

    /** The one persistent table. Index: {@code slot * blocksPerSlot + logicalBlock}. */
    private final int[] blockTable;

    private final BitSet leasedBlocks;

    /**
     * How many leases reference each block.
     *
     * <p>A block used to belong to exactly one lease, so a {@code BitSet} said everything. Prefix
     * sharing breaks that: two sequences that begin with the same tokens can read the same blocks,
     * and freeing on the first release would pull storage out from under the second. The bitset
     * still answers "is this block spoken for"; the counts answer "by how many".
     */
    private final int[] references;

    private final BitSet usedSlots;
    private final int totalBlocks;
    private final int blocksPerSlot;
    private final int maxSlots;
    private final int blockSizeTokens;
    private final long bytesPerBlock;

    /**
     * A table entry that names no block. Distinct from block 0, which is a real block.
     *
     * <p><b>Host-side only.</b> As a device index it would be an out-of-bounds negative write, and
     * inactive slots still execute the KV kernels every step. The store translates it to {@link
     * #scratchBlock()} when it publishes the table — see {@code TornadoKvStore.publishBlockTable}
     */
    public static final int UNMAPPED = -1;

    /** No slot: what a lease that could not be placed would carry. Never a valid table index. */
    public static final int NO_SLOT = -1;

    /**
     * The degenerate shape: as many slots as blocks, each able to hold the whole pool.
     *
     * <p>It asks no question about how leases will be sized, which is what makes it the right
     * default for a manager doing accounting with no device storage behind it, and for tests.
     * Production sizes the two dimensions explicitly through {@link KvCacheManager#sizedFor(int,
     * int, int, long)}, because the table it builds is {@code maxSlots * blocksPerSlot} entries and
     * this shape squares that.
     */
    public BlockPool(int totalBlocks, int blockSizeTokens, long bytesPerBlock) {
        this(totalBlocks, totalBlocks, totalBlocks, blockSizeTokens, bytesPerBlock);
    }

    /**
     * One further block is reserved beyond {@code totalBlocks} as the <b>scratch block</b>. It is
     * never leasable and never in the free set; it exists because an inactive slot still runs the
     * KV kernels every step and must write somewhere harmless. This is #129's arrangement, promoted
     * onto the pool.
     *
     * @param totalBlocks physical blocks the pool can hand out, excluding the scratch block
     * @param blocksPerSlot table entries reserved for one slot — a slot's maximum length
     * @param maxSlots how many sequences may be addressed at once
     * @param blockSizeTokens tokens one block holds
     * @param bytesPerBlock device bytes one block occupies across all layers
     */
    public BlockPool(
            int totalBlocks,
            int blocksPerSlot,
            int maxSlots,
            int blockSizeTokens,
            long bytesPerBlock) {
        if (totalBlocks <= 0) {
            throw new IllegalArgumentException(
                    "a pool needs at least one block, got " + totalBlocks);
        }
        if (blockSizeTokens <= 0) {
            throw new IllegalArgumentException(
                    "a block holds at least one token, got " + blockSizeTokens);
        }
        if (blocksPerSlot <= 0 || maxSlots <= 0) {
            throw new IllegalArgumentException(
                    "a pool needs at least one slot of at least one block, got "
                            + maxSlots
                            + " x "
                            + blocksPerSlot);
        }
        if ((long) maxSlots * blocksPerSlot > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("block table would overflow an int index");
        }
        this.totalBlocks = totalBlocks;
        this.blocksPerSlot = blocksPerSlot;
        this.maxSlots = maxSlots;
        this.blockTable = new int[maxSlots * blocksPerSlot];
        Arrays.fill(this.blockTable, UNMAPPED);
        this.leasedBlocks = new BitSet(totalBlocks);
        this.references = new int[totalBlocks];
        this.usedSlots = new BitSet(maxSlots);
        this.blockSizeTokens = blockSizeTokens;
        this.bytesPerBlock = bytesPerBlock;
    }

    /**
     * The block table itself, not a copy.
     *
     * <p>Deliberately the live array: its identity is part of the contract, and handing back a copy
     * would let a caller believe it had updated the table when it had not.
     */
    public int[] blockTable() {
        return blockTable;
    }

    public int totalBlocks() {
        return totalBlocks;
    }

    public int blocksPerSlot() {
        return blocksPerSlot;
    }

    public int maxSlots() {
        return maxSlots;
    }

    public int blockSizeTokens() {
        return blockSizeTokens;
    }

    public KvCapacity capacity() {
        return new KvCapacity(
                totalBlocks,
                totalBlocks - leasedBlocks.cardinality(),
                blockSizeTokens,
                bytesPerBlock);
    }

    public long bytesPerBlock() {
        return bytesPerBlock;
    }

    /** Claims a free slot, or {@link #NO_SLOT} when every slot is taken. */
    int reserveSlot() {
        int slot = usedSlots.nextClearBit(0);
        if (slot >= maxSlots) {
            return NO_SLOT;
        }
        usedSlots.set(slot);
        return slot;
    }

    /**
     * Reserves {@code count} blocks and maps them into {@code slot}'s range of the table.
     *
     * <p>Which physical blocks a slot gets is the pool's business, and deliberately not the
     * identity mapping any more: the whole point of the table is that the two are independent.
     *
     * @return the block indices reserved, in logical order
     * @throws IllegalStateException if the pool cannot satisfy the request
     */
    int[] reserve(int slot, int count) {
        if (count < 0) {
            throw new IllegalArgumentException(
                    "cannot reserve a negative number of blocks: " + count);
        }
        if (count > blocksPerSlot) {
            throw new IllegalStateException(
                    "a slot holds at most "
                            + blocksPerSlot
                            + " blocks, asked for "
                            + count
                            + ". The pool's slot length is fixed at"
                            + " construction, because changing it reshapes the table every graph captured");
        }
        int free = totalBlocks - leasedBlocks.cardinality();
        if (count > free) {
            throw new IllegalStateException(
                    DiagnosticCode.KV_POOL_EXHAUSTED.message(
                                    "KV pool exhausted: asked for " + count + " blocks, ")
                            + free
                            + " free of "
                            + totalBlocks
                            + ". Growing the pool invalidates captured graphs, so it is not done here");
        }
        int[] blocks = new int[count];
        int found = 0;
        for (int block = leasedBlocks.nextClearBit(0);
                found < count;
                block = leasedBlocks.nextClearBit(block + 1)) {
            leasedBlocks.set(block);
            references[block] = 1;
            blockTable[slot * blocksPerSlot + found] = block;
            blocks[found++] = block;
        }
        return blocks;
    }

    /** Returns blocks to the pool, unmaps the slot's range, and frees the slot. */
    void release(int slot, int[] blocks) {
        for (int block : blocks) {
            if (--references[block] <= 0) {
                references[block] = 0;
                leasedBlocks.clear(block);
            }
        }
        if (slot != NO_SLOT) {
            Arrays.fill(blockTable, slot * blocksPerSlot, (slot + 1) * blocksPerSlot, UNMAPPED);
            usedSlots.clear(slot);
        }
    }

    /** Whether a block is currently leased — a pinned block must never be handed out again. */
    public boolean isLeased(int block) {
        return block >= 0 && block < totalBlocks && leasedBlocks.get(block);
    }

    /**
     * The block inactive slots point at, and nothing else ever does.
     *
     * <p>It sits one past the leasable range, is never handed out, and never appears in a lease. A
     * slot with no sequence in it still executes the KV kernels each step [#129], so its writes
     * have to land somewhere that no live sequence reads. Pointing them at block 0 would corrupt
     * whoever holds it; leaving them {@link #UNMAPPED} would index out of bounds.
     */
    public int scratchBlock() {
        return totalBlocks;
    }

    /** Blocks the device storage must allocate: the leasable ones plus the scratch block. */
    public int allocatedBlocks() {
        return totalBlocks + 1;
    }

    /**
     * Evicts a block, or refuses if it is pinned by a live lease.
     *
     * @throws IllegalStateException if the block is leased — the caller may not evict it
     * @throws IllegalArgumentException if the block is the scratch block or out of range
     */
    public void evict(int block) {
        if (block == scratchBlock()) {
            throw new IllegalArgumentException(
                    "the scratch block is not evictable: it is what"
                            + " inactive slots write to, and every unmapped table entry points at it");
        }
        if (block < 0 || block >= totalBlocks) {
            throw new IllegalArgumentException(
                    "no such block: " + block + " (pool holds " + totalBlocks + ")");
        }
        if (leasedBlocks.get(block)) {
            throw new IllegalStateException(
                    "block "
                            + block
                            + " is pinned by "
                            + references[block]
                            + " live lease(s) and"
                            + " cannot be evicted. Re-pointing it would break replay of any captured graph"
                            + " holding it — as wrong output, not an error");
        }
        // Not leased: nothing to do. Eviction of a free block is a no-op rather than an error, so
        // that a caller sweeping a set of blocks need not first ask which of them are live.
    }

    /** The physical block backing a slot's logical block, or {@link #UNMAPPED}. */
    public int mapped(int slot, int logicalBlock) {
        return blockTable[slot * blocksPerSlot + logicalBlock];
    }

    /** Takes a reference on a block that is already held — the prefix cache is a holder too. */
    void retain(int block) {
        if (!leasedBlocks.get(block)) {
            throw new IllegalStateException("cannot retain free block " + block);
        }
        references[block]++;
    }

    /** Gives up one reference, freeing the block only when the last holder lets go. */
    void releaseOne(int block) {
        if (block < 0 || block >= totalBlocks) {
            return;
        }
        if (--references[block] <= 0) {
            references[block] = 0;
            leasedBlocks.clear(block);
        }
    }

    /** How many leases currently reference this block. Zero means free. */
    public int references(int block) {
        return block >= 0 && block < totalBlocks ? references[block] : 0;
    }

    /**
     * Points a slot's logical block at an existing physical block, and takes a reference on it.
     *
     * <p>This is what prefix sharing <i>is</i>: the same KV, read by two sequences, because they
     * begin with the same tokens at the same positions. The block is not copied and not re-reserved
     * — the second lease simply names it too, and the pool remembers that two of them do.
     *
     * @throws IllegalStateException if the block is not currently held by anyone; sharing a free
     *     block would hand out storage whose contents nothing guarantees
     */
    void share(int slot, int logicalBlock, int block) {
        if (block < 0 || block >= totalBlocks) {
            throw new IllegalArgumentException("no such block: " + block);
        }
        if (!leasedBlocks.get(block)) {
            throw new IllegalStateException(
                    "block "
                            + block
                            + " is free, so its contents are not"
                            + " anyone's to share. A prefix block stays referenced for as long as any lease"
                            + " names it — that is what the reference count is for");
        }
        references[block]++;
        blockTable[slot * blocksPerSlot + logicalBlock] = block;
    }
}
