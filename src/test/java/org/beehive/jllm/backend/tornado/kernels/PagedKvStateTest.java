package org.beehive.jllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.beehive.jllm.inference.state.LlamaState;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.llama.LlamaConfiguration;
import org.junit.Test;

/**
 * What a state allocates for KV, and the slot discipline it must keep.
 *
 * <p>Class A: allocates a small state, no device and no model file.
 */
public class PagedKvStateTest {

    private static final int DIM = 64;
    private static final int LAYERS = 3;
    private static final int HEADS = 8;
    private static final int KV_HEADS = 4;
    private static final int CONTEXT = 100; // deliberately not a multiple of the block size

    private static LlamaConfiguration config() {
        return new LlamaConfiguration(
                "FP16", DIM, 128, LAYERS, HEADS, KV_HEADS, 256, CONTEXT, 1e-5f, 500000.0f);
    }

    private static int kvDim() {
        return DIM * KV_HEADS / HEADS;
    }

    @Test
    public void aStateAllocatesAnIdentityMappedTableForOneSequence() {
        LlamaState state = new LlamaState(config(), -1);
        int blocksPerSeq = (CONTEXT + State.KV_BLOCK_SIZE - 1) / State.KV_BLOCK_SIZE;

        assertNotNull(state.workspace.wrapBlockTable);
        assertEquals(
                "the table is one sequence wide, which is why the slot must be 0",
                blocksPerSeq,
                state.workspace.wrapBlockTable.getSize());
        for (int b = 0; b < blocksPerSeq; b++) {
            assertEquals(
                    "logical block " + b + " maps to itself",
                    b,
                    state.workspace.wrapBlockTable.get(b));
        }
        assertEquals(State.KV_BLOCK_SIZE | (blocksPerSeq << 16), state.kvBlockCfg);
        assertEquals(LAYERS * State.KV_BLOCK_SIZE * kvDim(), state.kvBlockStride);
    }

    /**
     * The block-major cache is a permutation of the contiguous one plus at most {@code blockSize-1}
     * positions of padding — never a different amount of memory in any meaningful sense.
     */
    @Test
    public void theAllocationIsThePermutationPlusAtMostOneBlockOfPadding() {
        LlamaState paged = new LlamaState(config(), -1);
        int contiguous = CONTEXT * kvDim() * LAYERS;
        int slack = (State.KV_BLOCK_SIZE - 1) * kvDim() * LAYERS;

        // Whichever representation the storage selected (FP16 by default); the layout is the same.
        boolean fp16 = paged.usesFp16KeyValueCache();
        int keys =
                fp16
                        ? paged.workspace.wrapKeyCacheFP16.getSize()
                        : paged.workspace.wrapKeyCache.getSize();
        int values =
                fp16
                        ? paged.workspace.wrapValueCacheFP16.getSize()
                        : paged.workspace.wrapValueCache.getSize();
        assertTrue("never smaller", keys >= contiguous);
        assertTrue("padding is bounded by one block per layer", keys <= contiguous + slack);
        assertEquals(keys, values);
    }

    /**
     * The slot travels in {@code positionHolder[1]} so it can change between replays of a captured
     * graph. In this slice the bound table is one sequence wide, so the only slot that fits is 0 —
     * and the plans that drive the single-token path write element 0 only.
     */
    @Test
    public void slotIsCarriedAsDataAndIsZeroWhileTheTableIsStateLocal() {
        LlamaState state = new LlamaState(config(), -1);

        assertEquals("pos and slot", 2, state.workspace.positionHolder.getSize());
        state.workspace.positionHolder.init(0);
        state.workspace.positionHolder.set(0, 42);

        assertEquals(42, state.workspace.positionHolder.get(0));
        assertEquals(
                "a slot above 0 would index off the end of a one-sequence table",
                0,
                state.workspace.positionHolder.get(1));
        assertTrue(
                state.workspace.positionHolder.get(1) < state.workspace.wrapBlockTable.getSize());
    }
}
