package org.beehive.jitllm.golden;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.batch.TornadoBatchExecutor;
import org.beehive.jitllm.engine.LLMEngine;
import org.beehive.jitllm.engine.RequestHandle;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.runtime.kv.KvCacheManager;
import org.beehive.jitllm.runtime.kv.KvLease;
import org.beehive.jitllm.runtime.kv.KvStorage;
import org.beehive.jitllm.runtime.kv.KvStorageFactories;
import org.beehive.jitllm.runtime.kv.KvStorageRequest;

/**
 * Aggregate throughput through the engine at several batch widths.
 *
 * <p>The claim continuous batching makes is that aggregate tokens per second rises with B while
 * per-sequence latency stays usable. This measures the first half directly, through the production
 * engine rather than through #129's harness — which is the point of promoting the plan.
 *
 * <p>{@code -Dthroughput.model=… -Dthroughput.b=1,2,4,8 -Dthroughput.tokens=64}
 */
public final class BatchedEngineThroughput {

    public static void main(String[] args) throws Exception {
        Path modelPath = Path.of(System.getProperty("throughput.model"));
        int tokens = Integer.getInteger("throughput.tokens", 64);
        String[] widths = System.getProperty("throughput.b", "1,2,4").split(",");
        int contextLength = Integer.getInteger("throughput.ctx", 512);

        for (String width : widths) {
            int b = Integer.parseInt(width.trim());
            System.setProperty("jitllm.prefillBatchSize", String.valueOf(b));
            run(modelPath, b, tokens, contextLength);
        }
    }

    private static void run(Path modelPath, int batch, int tokensEach, int contextLength)
            throws Exception {
        Model model = ModelLoader.loadModel(modelPath, contextLength, true, true);
        int blockTokens = State.KV_BLOCK_SIZE;
        int blocksPerSlot = (contextLength + blockTokens - 1) / blockTokens;

        KvCacheManager manager = KvCacheManager.sizedFor(batch, contextLength, blockTokens, 4096);
        KvStorage store =
                KvStorageFactories.single()
                        .create(
                                new KvStorageRequest(
                                        batch * blocksPerSlot,
                                        blocksPerSlot,
                                        batch,
                                        blockTokens,
                                        model.configuration().numberOfLayers(),
                                        model.kvCacheDim(),
                                        false)); // the engine reads an FP32 pool only
        manager.attach(store);

        KvLease planLease = manager.acquire(blockTokens);
        State state = model.createNewState(planLease);

        try (TornadoBatchExecutor executor =
                new TornadoBatchExecutor(model, state, store, batch, blocksPerSlot)) {
            planLease.close();
            try (LLMEngine engine = new LLMEngine(model, manager, executor, batch, batch * 2)) {
                int begin = model.chatFormat().getBeginOfText();

                // Warm-up: compilation and any first-execution transfers must not land in the
                // timing.
                List<RequestHandle> warm = submit(engine, begin, batch, 8);
                drive(engine, warm);

                List<RequestHandle> handles = submit(engine, begin, batch, tokensEach);
                long start = System.nanoTime();
                int steps = drive(engine, handles);
                long elapsedNs = System.nanoTime() - start;

                int produced = 0;
                for (RequestHandle handle : handles) {
                    produced += handle.tokenCount();
                }
                double seconds = elapsedNs / 1e9;
                System.out.printf(
                        "B=%-2d  steps=%-4d  tokens=%-5d  aggregate=%.1f tok/s  "
                                + "per-sequence=%.1f tok/s%n",
                        batch, steps, produced, produced / seconds, produced / seconds / batch);
            }
        }
        manager.close();
    }

    private static List<RequestHandle> submit(LLMEngine engine, int token, int count, int budget) {
        List<RequestHandle> handles = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            handles.add(engine.addRequest(new int[] {token}, budget, null));
        }
        return handles;
    }

    private static int drive(LLMEngine engine, List<RequestHandle> handles) {
        int steps = 0;
        while (!allTerminal(handles) && steps < 4096) {
            engine.step();
            steps++;
        }
        return steps;
    }

    private static boolean allTerminal(List<RequestHandle> handles) {
        for (RequestHandle handle : handles) {
            if (!handle.isTerminal()) {
                return false;
            }
        }
        return true;
    }
}
