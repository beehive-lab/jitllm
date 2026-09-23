package org.beehive.jllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.HashSet;
import java.util.Set;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * This is the whole correctness argument for the addressing slice, written down. The walk replaces
 * {@code layer * contextLength * kvDim + pos * kvDim} in nine Llama single-token kernels; if it
 * addresses every (layer, position) exactly once, inside the allocation, in the layout the kernels
 * assume, then a golden that moves is a numerics bug and not an addressing one.
 */
public class KvBlockAddressTest {

    private static final int BLOCK_SIZE = 16;
    private static final int NUM_LAYERS = 4;
    private static final int KV_DIM = 8;
    private static final int CONTEXT = 64;
    private static final int BLOCKS_PER_SEQ = (CONTEXT + BLOCK_SIZE - 1) / BLOCK_SIZE;
    private static final int BLOCK_CFG = BLOCK_SIZE | (BLOCKS_PER_SEQ << 16);
    private static final int BLOCK_STRIDE = NUM_LAYERS * BLOCK_SIZE * KV_DIM;

    private static IntArray identityTable() {
        IntArray table = new IntArray(BLOCKS_PER_SEQ);
        for (int b = 0; b < BLOCKS_PER_SEQ; b++) {
            table.set(b, b);
        }
        return table;
    }

    /** The layout as documented, written out independently of the implementation. */
    private static int reference(int physBlock, int layer, int pos) {
        return physBlock * (NUM_LAYERS * BLOCK_SIZE * KV_DIM)
                + layer * (BLOCK_SIZE * KV_DIM)
                + (pos % BLOCK_SIZE) * KV_DIM;
    }

    @Test
    public void packedConfigRoundTrips() {
        assertEquals(BLOCK_SIZE, KvBlockAddress.blockSize(BLOCK_CFG));
        assertEquals(BLOCKS_PER_SEQ, KvBlockAddress.maxBlocksPerSlot(BLOCK_CFG));
    }

    @Test
    public void offsetMatchesTheDocumentedLayout() {
        IntArray table = identityTable();
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            int layerOff = KvBlockAddress.layerOffset(layer, KV_DIM, BLOCK_CFG);
            for (int pos = 0; pos < CONTEXT; pos++) {
                int expected = reference(pos / BLOCK_SIZE, layer, pos);
                int actual =
                        KvBlockAddress.offset(
                                table, 0, pos, layerOff, KV_DIM, BLOCK_CFG, BLOCK_STRIDE);
                assertEquals("layer " + layer + " pos " + pos, expected, actual);
            }
        }
    }

    /**
     * No two (layer, position) pairs share a row, and every row lands inside the allocation. This
     * is what the contiguous index gave for free and what an indirection can lose.
     */
    @Test
    public void everyPositionGetsItsOwnRowInsideTheAllocation() {
        IntArray table = identityTable();
        int allocation = BLOCKS_PER_SEQ * NUM_LAYERS * BLOCK_SIZE * KV_DIM;
        Set<Integer> seen = new HashSet<>();
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            int layerOff = KvBlockAddress.layerOffset(layer, KV_DIM, BLOCK_CFG);
            for (int pos = 0; pos < CONTEXT; pos++) {
                int off =
                        KvBlockAddress.offset(
                                table, 0, pos, layerOff, KV_DIM, BLOCK_CFG, BLOCK_STRIDE);
                assertTrue("row starts inside the allocation", off >= 0);
                assertTrue("row ends inside the allocation", off + KV_DIM <= allocation);
                assertTrue("layer " + layer + " pos " + pos + " collided at " + off, seen.add(off));
            }
        }
        assertEquals("every (layer, position) is addressed", NUM_LAYERS * CONTEXT, seen.size());
    }

    @Test
    public void redirectingALogicalBlockMovesExactlyThatBlocksPositions() {
        IntArray table = identityTable();
        int layerOff = KvBlockAddress.layerOffset(1, KV_DIM, BLOCK_CFG);
        int before = KvBlockAddress.offset(table, 0, 20, layerOff, KV_DIM, BLOCK_CFG, BLOCK_STRIDE);

        table.set(1, 3); // logical block 1 now lives in physical block 3
        int after = KvBlockAddress.offset(table, 0, 20, layerOff, KV_DIM, BLOCK_CFG, BLOCK_STRIDE);

        assertEquals(reference(1, 1, 20), before);
        assertEquals(reference(3, 1, 20), after);
        assertEquals(
                "a position in another block is untouched",
                reference(0, 1, 4),
                KvBlockAddress.offset(table, 0, 4, layerOff, KV_DIM, BLOCK_CFG, BLOCK_STRIDE));
    }

    @Test
    public void slotStridesByMaxBlocksPerSlot() {
        int slots = 3;
        IntArray wide = new IntArray(slots * BLOCKS_PER_SEQ);
        for (int i = 0; i < wide.getSize(); i++) {
            wide.set(i, i);
        }
        int layerOff = KvBlockAddress.layerOffset(0, KV_DIM, BLOCK_CFG);
        for (int slot = 0; slot < slots; slot++) {
            int off =
                    KvBlockAddress.offset(wide, slot, 0, layerOff, KV_DIM, BLOCK_CFG, BLOCK_STRIDE);
            assertEquals("slot " + slot, reference(slot * BLOCKS_PER_SEQ, 0, 0), off);
        }
    }
}
