package org.beehive.jllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.beehive.jllm.backend.tornado.TensorCoreSupport;
import org.beehive.jllm.backend.tornado.batch.TornadoBatchExecutor;
import org.beehive.jllm.engine.LLMEngine;
import org.beehive.jllm.engine.RequestHandle;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.loader.ModelLoader;
import org.beehive.jllm.runtime.kv.KvCacheManager;
import org.beehive.jllm.runtime.kv.KvLease;
import org.beehive.jllm.runtime.kv.KvStorage;
import org.beehive.jllm.runtime.kv.KvStorageFactories;
import org.beehive.jllm.runtime.kv.KvStorageRequest;
import org.junit.Test;

/**
 * The continuous-batch engine with an FP16 key/value pool against the same engine with an FP32 one:
 * two concurrent requests with prompts long enough to fill several pool blocks, then 64 greedy
 * tokens each.
 *
 * <p>The engine samples on the device and returns token ids only, so this compares the greedy
 * streams rather than logits — the weaker evidence, and stated as such. The single-session paths
 * have the row-by-row logit comparison ({@link KvPrecisionHarness}); the engine's decode kernels
 * are FP16-storage twins of its FP32 ones.
 */
public class EngineFp16KvAccelTest {

    private static final int BATCH = 2;
    private static final int CONTEXT_LENGTH = 1024;
    private static final int BLOCK_TOKENS = State.KV_BLOCK_SIZE;
    private static final int NEW_TOKENS = 64;

    @Test
    public void llamaF16() throws Exception {
        check(Fixture.LLAMA_3_2_1B_F16);
    }

    @Test
    public void qwen3F16() throws Exception {
        check(Fixture.QWEN3_0_6B_F16);
    }

    private static void check(Fixture fixture) throws Exception {
        Path file = GoldenFixture.locate(fixture);
        assumeTrue("environment absent: " + GoldenFixture.absentMessage(fixture), file != null);
        assumeTrue(
                "the engine needs tensor-core MMA", TensorCoreSupport.isTensorCoreCapableBackend());
        String previousBatch = System.getProperty("jllm.prefillBatchSize");
        System.setProperty("jllm.prefillBatchSize", String.valueOf(BATCH));
        try {
            Model model = ModelLoader.loadModel(file, CONTEXT_LENGTH, true, true);
            List<Integer> longPrompt = KvPrecisionHarness.longPrompt(model, 300);
            int[] a = longPrompt.stream().mapToInt(Integer::intValue).toArray();
            int[] b =
                    KvPrecisionHarness.longPrompt(model, 150).stream()
                            .mapToInt(Integer::intValue)
                            .toArray();
            List<List<Integer>> fp32 = run(model, false, a, b);
            List<List<Integer>> fp16 = run(model, true, a, b);
            for (int r = 0; r < 2; r++) {
                List<Integer> x = fp32.get(r);
                List<Integer> y = fp16.get(r);
                int n = Math.min(x.size(), y.size());
                int firstDivergence = n;
                int agree = 0;
                for (int i = 0; i < n; i++) {
                    if (x.get(i).equals(y.get(i))) agree++;
                    else if (firstDivergence == n) firstDivergence = i;
                }
                System.out.printf(
                        "[kv-precision] engine %s request %d: %d tokens, first divergence %s,"
                                + " agreement %.3f%n",
                        fixture,
                        r,
                        n,
                        firstDivergence == n ? "none" : Integer.toString(firstDivergence),
                        n == 0 ? 0.0 : (double) agree / n);
                // A request may end at its stop token before NEW_TOKENS; both runs must then end
                // together unless they diverged first.
                assertTrue("request " + r + " produced too few tokens: " + n, n >= 16);
                if (firstDivergence == n) {
                    assertEquals(
                            "request " + r + " ended at different lengths", x.size(), y.size());
                }
                assertTrue(
                        "request " + r + " diverged at token " + firstDivergence,
                        firstDivergence >= 16);
            }
        } finally {
            if (previousBatch == null) {
                System.clearProperty("jllm.prefillBatchSize");
            } else {
                System.setProperty("jllm.prefillBatchSize", previousBatch);
            }
        }
    }

    private static List<List<Integer>> run(Model model, boolean fp16, int[] a, int[] b)
            throws Exception {
        int blocksPerSlot = (CONTEXT_LENGTH + BLOCK_TOKENS - 1) / BLOCK_TOKENS;
        KvCacheManager manager =
                KvCacheManager.sizedFor(BATCH, CONTEXT_LENGTH, BLOCK_TOKENS, fp16 ? 2048 : 4096);
        KvStorage store =
                KvStorageFactories.single()
                        .create(
                                new KvStorageRequest(
                                        BATCH * blocksPerSlot,
                                        blocksPerSlot,
                                        BATCH,
                                        BLOCK_TOKENS,
                                        model.configuration().numberOfLayers(),
                                        model.kvCacheDim(),
                                        fp16));
        manager.attach(store);
        KvLease planLease = manager.acquire(BLOCK_TOKENS);
        State state = model.createNewState(planLease);
        assertEquals(
                "the plan state binds the pool it was given", fp16, state.usesFp16KeyValueCache());
        List<List<Integer>> out = new ArrayList<>();
        try (TornadoBatchExecutor executor =
                new TornadoBatchExecutor(model, state, store, BATCH, blocksPerSlot)) {
            planLease.close();
            try (LLMEngine engine = new LLMEngine(model, manager, executor, BATCH, 8)) {
                RequestHandle ra = engine.addRequest(a, NEW_TOKENS, null);
                RequestHandle rb = engine.addRequest(b, NEW_TOKENS, null);
                int steps = 0;
                while ((!ra.isTerminal() || !rb.isTerminal()) && steps < 4096) {
                    engine.step();
                    steps++;
                }
                out.add(List.copyOf(ra.tokens()));
                out.add(List.copyOf(rb.tokens()));
            }
        }
        manager.close();
        return out;
    }
}
