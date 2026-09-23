package org.beehive.jitllm.inference.state;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import org.beehive.jitllm.model.qwen35.Qwen35Configuration;
import org.junit.After;
import org.junit.Test;

/**
 * What a {@code qwen35} session allocates, on each path.
 *
 * <p>Two things here are unlike every other family and both are about size rather than correctness
 * — which is why they need a test at all. Only one block in four attends, so a key/value store
 * sized by the block count would be four times larger than the model uses; and the recurrent layers
 * hold state that no other family has, which must exist on whichever path is running and must start
 * at zero on both.
 */
public class Qwen35StateTest {

    /** Qwen3.8-27B's shape, at a context small enough to allocate in a test. */
    private static Qwen35Configuration config() {
        return new Qwen35Configuration(
                "Q8_0", 5120, 17408, 64, 1, 24, 4, 256, 256, 4, 4, 128, 16, 48, 6144, 64, 248320,
                262144, /* contextLength */ 256, 1e-6f, 1e7f);
    }

    @After
    public void clearTheDeviceProperty() {
        System.clearProperty("use.tornadovm");
    }

    @Test
    public void theHostPathAllocatesNoDeviceArrays() {
        System.setProperty("use.tornadovm", "false");
        Qwen35State state = new Qwen35State(config(), -1);

        assertNull("device recurrent state", state.workspace.wrapConvState);
        assertNull("device key cache", state.workspace.wrapKeyCache);
        assertEquals("no block layout", 0, state.kvBlockCfg);
    }

    @Test
    public void keyValueStorageCoversOnlyTheBlocksThatAttend() {
        System.setProperty("use.tornadovm", "true");
        Qwen35Configuration config = config();
        Qwen35State state = new Qwen35State(config, -1);

        // 16 attending trunk layers plus the one MTP block.
        assertEquals(17, config.keyValueLayerCount());

        int blockSize = State.KV_BLOCK_SIZE;
        int blocksPerSeq = (config.contextLength() + blockSize - 1) / blockSize;
        int expected = blocksPerSeq * config.keyValueLayerCount() * blockSize * config.kvDim();
        assertEquals(
                "the store is sized by the blocks that write to it",
                expected,
                state.usesFp16KeyValueCache()
                        ? state.workspace.wrapKeyCacheFP16.getSize()
                        : state.workspace.wrapKeyCache.getSize());

        // Sized by the block count instead, it would be nearly four times this.
        int naive = blocksPerSeq * config.numberOfBlocks() * blockSize * config.kvDim();
        assertTrue("the sparse sizing is the point", expected * 3 < naive);
    }

    @Test
    public void theDenseIndicesCoverEachKindExactlyOnce() {
        Qwen35Configuration config = config();

        boolean[] seenKv = new boolean[config.keyValueLayerCount()];
        boolean[] seenRecurrent = new boolean[config.recurrentLayerCount()];
        for (int l = 0; l < config.numberOfBlocks(); l++) {
            int kv = config.keyValueLayerIndex(l);
            int recurrent = config.recurrentLayerIndex(l);
            // Every block is one kind or the other, never both and never neither.
            assertTrue("block " + l + " is neither", kv >= 0 || recurrent >= 0);
            assertTrue("block " + l + " is both", kv < 0 || recurrent < 0);
            if (kv >= 0) {
                assertTrue("kv index " + kv + " reused", !seenKv[kv]);
                seenKv[kv] = true;
            } else {
                assertTrue("recurrent index " + recurrent + " reused", !seenRecurrent[recurrent]);
                seenRecurrent[recurrent] = true;
            }
        }
        for (boolean seen : seenKv) {
            assertTrue("a key/value index was never claimed", seen);
        }
        for (boolean seen : seenRecurrent) {
            assertTrue("a recurrent index was never claimed", seen);
        }
    }

    @Test
    public void theRecurrentStateExistsOnBothPathsAndStartsAtZero() {
        System.setProperty("use.tornadovm", "true");
        Qwen35Configuration config = config();
        Qwen35State state = new Qwen35State(config, -1);

        assertNotNull(state.workspace.wrapConvState);
        assertEquals(
                config.recurrentLayerCount() * config.convStateSize(),
                state.workspace.wrapConvState.getSize());
        assertEquals(
                config.recurrentLayerCount() * config.deltaNetStateSize(),
                state.workspace.wrapDeltaState.getSize());

        // A recurrence has no position mask, so whatever was in the allocation would be read as
        // the sequence's own history.
        for (int i = 0; i < state.workspace.wrapDeltaState.getSize(); i += 4099) {
            assertEquals("delta state at " + i, 0f, state.workspace.wrapDeltaState.get(i), 0f);
        }
        for (int i = 0; i < state.workspace.wrapConvState.getSize(); i += 997) {
            assertEquals("conv state at " + i, 0f, state.workspace.wrapConvState.get(i), 0f);
        }
    }

    @Test
    public void aResetClearsBothRepresentationsOfTheRecurrentState() {
        System.setProperty("use.tornadovm", "true");
        Qwen35Configuration config = config();
        Qwen35State state = new Qwen35State(config, -1);

        state.workspace.wrapDeltaState.set(17, 1.5f);
        state.workspace.wrapConvState.set(3, -2.5f);
        state.deltaState[0].setFloat(9, 4.0f);

        state.resetSequenceState();

        assertEquals(0f, state.workspace.wrapDeltaState.get(17), 0f);
        assertEquals(0f, state.workspace.wrapConvState.get(3), 0f);
        assertEquals(0f, state.deltaState[0].getFloat(9), 0f);
    }
}
