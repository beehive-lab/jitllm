package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.runtime.kv.KvCacheManager;
import org.beehive.jitllm.runtime.kv.KvLease;
import org.beehive.jitllm.runtime.kv.KvStorage;
import org.beehive.jitllm.runtime.kv.KvStorageFactories;
import org.beehive.jitllm.runtime.kv.KvStorageRequest;
import org.junit.Test;

/**
 * The requirement reads: "evict a block while a captured graph holds it, and assert the failure is
 * caught rather than silently producing wrong output". That is two claims, and this test makes
 * both, because either alone proves less than it looks:
 *
 * <ol>
 *   <li><b>The manager refuses</b> to evict a block a live lease pins, while a real plan is bound
 *       to that pool. Without the second claim this only proves a {@code BitSet} works.
 *   <li><b>Re-pointing is detected, not absorbed.</b> The reason pinning exists is [C1]: a captured
 *       graph holds device <i>addresses</i>, so handing a bound buffer a different array is wrong
 *       output rather than an error whenever {@code tornado.recover.bailout} is at its default. The
 *       negative control shows the difference is observable — which is what makes the refusal above
 *       worth having.
 * </ol>
 *
 * <p>Runs under {@code -Dtornado.recover.bailout=False} (capability C4), as the accel profile sets.
 */
public class PagedPoolEvictionAccelTest {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    private static final int CONTEXT_LENGTH = 512;
    private static final int BLOCK_TOKENS = State.KV_BLOCK_SIZE;

    /**
     * One model, one plan, both halves — deliberately.
     *
     * <p>Each GPU plan in this suite holds its own device copy of the weights, and the Class B
     * budget is sized for the plans that already exist; a test that stands up two takes an
     * unrelated test down with it, and the failure lands on whichever ran last rather than on the
     * one that overspent. {@code MultiSessionAccelTest} records the same lesson. So the refusal and
     * its negative control share a plan.
     */
    @Test
    public void aPinnedBlockIsRefusedAndRepointingItWouldHaveMattered() throws Exception {
        Path modelPath = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_Q8_0);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — "
                            + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_Q8_0));
            assumeTrue("environment absent", false);
        }
        Model model = ModelLoader.loadModel(modelPath, CONTEXT_LENGTH, true, true);

        int blocksPerSlot = (CONTEXT_LENGTH + BLOCK_TOKENS - 1) / BLOCK_TOKENS;
        KvCacheManager manager = KvCacheManager.sizedFor(1, CONTEXT_LENGTH, BLOCK_TOKENS, 4096);
        KvStorage store =
                KvStorageFactories.single()
                        .create(
                                new KvStorageRequest(
                                        blocksPerSlot,
                                        blocksPerSlot,
                                        1,
                                        BLOCK_TOKENS,
                                        model.configuration().numberOfLayers(),
                                        model.kvCacheDim(),
                                        false)); // the engine reads an FP32 pool only
        manager.attach(store);

        KvLease lease = manager.acquire(CONTEXT_LENGTH);
        State state = model.createNewState(lease);
        TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
        try {
            int token = beginToken(model);
            // Two positions, so there is KV history for a mapping change to matter to.
            org.beehive.jitllm.backend.tornado.TornadoForwardPass.forward(
                    model, state, token, 0, plan);
            org.beehive.jitllm.backend.tornado.TornadoForwardPass.forward(
                    model, state, token, 1, plan);
            float[] before = firstLogits(state);

            // ── 1. the refusal ────────────────────────────────────────────────────────────────
            int held = manager.pool().mapped(lease.slot(), 0);
            assertTrue("the lease pins it", manager.pool().isLeased(held));

            IllegalStateException refused =
                    assertThrows(IllegalStateException.class, () -> manager.evict(held));
            assertTrue(refused.getMessage(), refused.getMessage().contains("pinned"));
            assertTrue("and it stays leased", manager.pool().isLeased(held));

            // The scratch block is not evictable either: every inactive slot points at it.
            assertThrows(
                    IllegalArgumentException.class,
                    () -> manager.evict(manager.pool().scratchBlock()));

            // ── 2. the negative control: the refusal is guarding something ───────────────────
            // Re-point logical block 0 at another physical block — what an eviction under a live
            // lease does — and show the answer moves. Two things must hold: the kernels read the
            // pool through the table, and the table reaches the device after the first execution
            // (it is uploaded EVERY_EXECUTION for exactly that reason). Bound once, this
            // would pass by accident and the engine's admissions would be equally invisible.
            int[] table = manager.pool().blockTable();
            int original = table[lease.slot() * blocksPerSlot];
            int other = (original + 1) % blocksPerSlot;
            assertNotEquals("the re-point must be a different block", original, other);
            table[lease.slot() * blocksPerSlot] = other;
            store.publishBlockTable(table);

            org.beehive.jitllm.backend.tornado.TornadoForwardPass.forward(
                    model, state, token, 1, plan);
            float[] after = firstLogits(state);

            assertEquals(before.length, after.length);
            boolean changed = false;
            for (int i = 0; i < before.length && !changed; i++) {
                changed = before[i] != after[i];
            }
            assertTrue(
                    "re-pointing a leased block must change the answer — if it does not, either"
                            + " the kernels are not reading the table or the table never reached the"
                            + " device, and pinning guards nothing",
                    changed);

            table[lease.slot() * blocksPerSlot] = original;
            store.publishBlockTable(table);
            System.out.println(
                    "[EVICTION] pinned block refused; re-pointing it moves the logits, "
                            + "so the refusal guards something real");
        } finally {
            plan.freeTornadoExecutionPlan();
            lease.close();
            manager.close();
        }
    }

    private static float[] firstLogits(State state) {
        int n = Math.min(64, state.workspace.wrapLogits.getSize());
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = state.workspace.wrapLogits.get(i);
        }
        return out;
    }

    private static int beginToken(Model model) {
        return model.shouldAddBeginOfText()
                ? model.chatFormat().getBeginOfText()
                : model.tokenizer().getSpecialTokens().values().iterator().next();
    }
}
