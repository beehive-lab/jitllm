package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.backend.tornado.batch.TornadoBatchExecutor;
import org.beehive.jitllm.engine.LLMEngine;
import org.beehive.jitllm.engine.RequestHandle;
import org.beehive.jitllm.engine.RequestState;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.runtime.kv.KvCacheManager;
import org.beehive.jitllm.runtime.kv.KvLease;
import org.beehive.jitllm.runtime.kv.KvStorage;
import org.beehive.jitllm.runtime.kv.KvStorageFactories;
import org.beehive.jitllm.runtime.kv.KvStorageRequest;
import org.junit.Test;

public class BatchedEngineAccelTest {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    private static final int BATCH = 2;
    private static final int CONTEXT_LENGTH = 256;
    private static final int BLOCK_TOKENS = State.KV_BLOCK_SIZE;

    @Test
    public void twoSequencesDecodeInOneBatchThroughTheEngine() throws Exception {
        Path modelPath = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_F16);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — "
                            + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_F16));
            assumeTrue("environment absent", false);
        }
        if (!TensorCoreSupport.isTensorCoreCapableBackend()) {
            // Metal parity task 9: TornadoVM lowers the MMA intrinsics (gemmMMAQKV and friends)
            // only on the PTX/CUDA backend — an already-decided, documented capability gap
            // (, "Already decided, and not open":
            // "TENSOR_CORE_MMA is CUDA-only. not a Metal gap"), not something to characterise
            // fresh here. Without this gate the request bails out deep inside TornadoVM's sketcher
            // with an opaque TornadoInternalError instead of a named, actionable skip.
            System.out.println(
                    "[SKIP] device lacks DeviceCapability.TENSOR_CORE_MMA — TornadoVM"
                            + " only lowers the MMA batch-prefill kernels on the PTX/CUDA backend");
            assumeTrue("device lacks DeviceCapability.TENSOR_CORE_MMA", false);
        }

        // The batched buffers on State are allocated only when a batch is asked for, and the plan
        // is built for exactly B rows.
        String previousBatch = System.getProperty("jitllm.prefillBatchSize");
        System.setProperty("jitllm.prefillBatchSize", String.valueOf(BATCH));
        try {
            Model model = ModelLoader.loadModel(modelPath, CONTEXT_LENGTH, true, true);
            assertTrue(
                    "the engine only accepts families that can share KV storage",
                    model.supportsSharedKvStorage());

            int blocksPerSlot = (CONTEXT_LENGTH + BLOCK_TOKENS - 1) / BLOCK_TOKENS;
            KvCacheManager manager =
                    KvCacheManager.sizedFor(BATCH, CONTEXT_LENGTH, BLOCK_TOKENS, 4096);
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
                                            false)); // the engine reads an FP32 pool only
            manager.attach(store);

            // One state, one lease: the batched plan binds the shared pool, and the per-slot
            // addressing comes from the block table rather than from separate states.
            KvLease planLease = manager.acquire(BLOCK_TOKENS);
            State state = model.createNewState(planLease);

            try (TornadoBatchExecutor executor =
                    new TornadoBatchExecutor(model, state, store, BATCH, blocksPerSlot)) {
                planLease.close(); // the plan holds the buffers now; the slot goes back to the pool

                try (LLMEngine engine = new LLMEngine(model, manager, executor, BATCH, 8)) {
                    int begin = model.chatFormat().getBeginOfText();
                    List<Integer> firstUser =
                            model.chatFormat()
                                    .encodeMessage(
                                            new ChatFormat.Message(
                                                    ChatFormat.Role.USER, "Name one colour."));

                    List<Integer> received = new ArrayList<>();
                    // Prepared tokens, as the engine takes them: ids from the model's own
                    // template, never a rendered string.
                    RequestHandle a = engine.addRequest(new int[] {begin}, 24, received::add);
                    RequestHandle b =
                            engine.addRequest(new int[] {begin, firstUser.get(0)}, 24, null);

                    int steps = 0;
                    while ((!a.isTerminal() || !b.isTerminal()) && steps < 64) {
                        engine.step();
                        steps++;
                    }

                    assertTrue("both sequences advanced", a.tokenCount() > 0 && b.tokenCount() > 0);
                    assertTrue(
                            "the callback saw every token of its own request",
                            received.size() == a.tokenCount());
                    assertNotEquals(
                            "two slots decoded independently, not in lockstep",
                            a.tokens(),
                            b.tokens());
                    assertTrue(
                            "and both reached a terminal state", a.isTerminal() && b.isTerminal());
                    assertTrue(a.state() == RequestState.COMPLETED);

                    System.out.println(
                            "[BATCHED ENGINE] B="
                                    + BATCH
                                    + " steps="
                                    + steps
                                    + " tokens="
                                    + a.tokenCount()
                                    + "/"
                                    + b.tokenCount());
                }
            }
            assertEquals("every lease came back", 0, manager.liveLeases());
            manager.close();
        } finally {
            if (previousBatch == null) {
                System.clearProperty("jitllm.prefillBatchSize");
            } else {
                System.setProperty("jitllm.prefillBatchSize", previousBatch);
            }
        }
    }
}
