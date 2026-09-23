package org.beehive.jitllm.inference.state;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.runtime.kv.BlockPool;
import org.beehive.jitllm.runtime.kv.KvCacheManager;
import org.beehive.jitllm.runtime.kv.KvLease;
import org.beehive.jitllm.runtime.kv.KvStorage;
import org.beehive.jitllm.runtime.kv.KvStorageFactories;
import org.beehive.jitllm.runtime.kv.KvStorageRequest;
import org.junit.Test;

/**
 * Class A: allocates a small pool on the host through the backend's factory; no model file and no
 * kernel runs. The arrays are TornadoVM native arrays, which allocate without a device.
 */
public class KvBindingTest {

    private static final int DIM = 64;
    private static final int LAYERS = 3;
    private static final int HEADS = 8;
    private static final int KV_HEADS = 4;
    private static final int CONTEXT = 100;

    private static LlamaConfiguration config() {
        return new LlamaConfiguration(
                "FP16", DIM, 128, LAYERS, HEADS, KV_HEADS, 256, CONTEXT, 1e-5f, 500000.0f);
    }

    private static int kvDim() {
        return DIM * KV_HEADS / HEADS;
    }

    /** Acceptance 1 — no lease, and a lease without storage, both keep private allocation. */
    @Test
    public void aStateWithoutBackendStorageStillAllocatesItsOwnArrays() {
        LlamaState noLease = new LlamaState(config(), -1, null);
        assertNotNull(
                "no lease means no backend storage was requested, which is not an error",
                ownKeyCache(noLease));
        assertNotNull(noLease.workspace.wrapBlockTable);

        // A CPU-only runtime: a real lease, no storage attached. Valid, and needing no CPU binder.
        int blocksPerSlot = (CONTEXT + State.KV_BLOCK_SIZE - 1) / State.KV_BLOCK_SIZE;
        try (KvCacheManager manager =
                new KvCacheManager(
                        new BlockPool(
                                blocksPerSlot, blocksPerSlot, 1, State.KV_BLOCK_SIZE, 1024))) {
            KvLease lease = manager.acquire(CONTEXT);
            assertNull("the premise of this case", lease.storage());
            LlamaState state = new LlamaState(config(), -1, lease);
            assertNotNull(ownKeyCache(state));
            lease.close();
        }
    }

    /** Acceptance 2 — leased storage binds the pool's own arrays and its configuration values. */
    @Test
    public void leasedStorageBindsTheStoresArrayIdentitiesAndConfiguration() {
        int blocksPerSlot = (CONTEXT + State.KV_BLOCK_SIZE - 1) / State.KV_BLOCK_SIZE;
        KvStorage store =
                KvStorageFactories.single()
                        .create(
                                new KvStorageRequest(
                                        blocksPerSlot,
                                        blocksPerSlot,
                                        1,
                                        State.KV_BLOCK_SIZE,
                                        LAYERS,
                                        kvDim(),
                                        false));
        try (KvCacheManager manager =
                new KvCacheManager(
                        new BlockPool(
                                blocksPerSlot,
                                blocksPerSlot,
                                1,
                                State.KV_BLOCK_SIZE,
                                store.bytesPerBlock()))) {
            manager.attach(store);
            KvLease lease = manager.acquire(CONTEXT);
            LlamaState state = new LlamaState(config(), -1, lease);

            // Identity, not equality: sharing a pool means pointing at the same arrays. A copy
            // would pass an equality check and share nothing.
            assertSame(
                    "the key pool is the store's, not a copy",
                    arrayOf(store, "keyPool"),
                    state.workspace.wrapKeyCache);
            assertSame(arrayOf(store, "valuePool"), state.workspace.wrapValueCache);
            assertSame(arrayOf(store, "blockTable"), state.workspace.wrapBlockTable);
            assertNull(
                    "FP32 was requested, so the half-precision pair is not allocated",
                    state.workspace.wrapKeyCacheFP16);

            // The stride is the distance between two blocks: numLayers * blockSize * kvDim.
            assertEquals(
                    "the block stride is the store's, not recomputed by the state",
                    LAYERS * State.KV_BLOCK_SIZE * kvDim(),
                    state.kvBlockStride);
            // The config packs the block size in the low half and blocksPerSlot in the high half.
            assertEquals(
                    "the block config is the store's",
                    State.KV_BLOCK_SIZE | (blocksPerSlot << 16),
                    state.kvBlockCfg);
            lease.close();
        }
        store.close();
    }

    /** Acceptance 3 — storage nothing claims throws, and names what is wrong. */
    @Test
    public void storageNoBinderClaimsThrowsRatherThanAllocatingPrivately() {
        int blocksPerSlot = (CONTEXT + State.KV_BLOCK_SIZE - 1) / State.KV_BLOCK_SIZE;
        try (KvCacheManager manager =
                new KvCacheManager(
                        new BlockPool(
                                blocksPerSlot, blocksPerSlot, 1, State.KV_BLOCK_SIZE, 1024))) {
            manager.attach(new ForeignStorage());
            KvLease lease = manager.acquire(CONTEXT);

            IllegalStateException thrown =
                    assertThrows(
                            IllegalStateException.class, () -> new LlamaState(config(), -1, lease));
            // Falling back here would give correct output, more memory and no explanation on a
            // machine that explicitly asked for a shared pool.
            assertTrue(
                    thrown.getMessage(),
                    thrown.getMessage().contains(ForeignStorage.class.getName()));
            assertTrue(
                    thrown.getMessage(),
                    thrown.getMessage().contains("no compatible backend binding exists"));
            lease.close();
        }
    }

    private static Object arrayOf(KvStorage store, String accessor) {
        try {
            return store.getClass().getMethod(accessor).invoke(store);
        } catch (ReflectiveOperationException e) {
            throw new AssertionError("the backend store lost its " + accessor + "() accessor", e);
        }
    }

    private static final class ForeignStorage implements KvStorage {
        @Override
        public void publishBlockTable(int[] blockTable) {}

        @Override
        public int blockSizeTokens() {
            return State.KV_BLOCK_SIZE;
        }

        @Override
        public int blocksPerSlot() {
            return 1;
        }

        @Override
        public long bytesPerBlock() {
            return 1024;
        }

        @Override
        public void close() {}
    }

    /**
     * The key array a private state allocated, in whichever representation its storage selects —
     * and only that one: FP16 is the default, and no FP32 pair is kept beside it.
     */
    private static Object ownKeyCache(State state) {
        if (state.usesFp16KeyValueCache()) {
            assertNull("no unused FP32 pair beside the FP16 cache", state.workspace.wrapKeyCache);
            return state.workspace.wrapKeyCacheFP16;
        }
        return state.workspace.wrapKeyCache;
    }
}
