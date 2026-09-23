package org.beehive.jitllm.golden;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.batch.TornadoBatchExecutor;
import org.beehive.jitllm.engine.LLMEngine;
import org.beehive.jitllm.engine.RequestHandle;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.runtime.kv.KvCacheManager;
import org.beehive.jitllm.runtime.kv.KvLease;
import org.beehive.jitllm.runtime.kv.KvStorage;
import org.beehive.jitllm.runtime.kv.KvStorageFactories;
import org.beehive.jitllm.runtime.kv.KvStorageRequest;

/**
 * The promoted path on #129's workload, so the two numbers mean the same thing.
 *
 * <p>Matching the workload took more care than expected, and the mismatches are worth naming
 * because each would have produced a confident wrong comparison:
 *
 * <ul>
 *   <li>#129's requests stop at the <b>stop token</b>, not at the configured token count — that is
 *       an upper bound. Its 64 requests generate about eight tokens each, not sixty-four.
 *   <li>Its reported {@code aggregateTokensPerSecond} is {@code timedGen / sec}: tokens from the
 *       <b>timed steps</b> divided by the <b>whole</b> wall clock, warm-up included. The two
 *       disagree by however long warm-up took.
 *   <li>Its run includes prefill; a decode-only measurement is a different quantity.
 * </ul>
 *
 * <p>So this drives the engine with the same prompt, the same request count, the same batch width
 * and real stop tokens, and reports tokens over the whole wall clock — the quantity that is
 * comparable.
 *
 * <p>{@code -Dprobe.model=… -Dprobe.requests=64 -Dprobe.b=16}
 */
public final class EngineVsReference129 {

    public static void main(String[] args) throws Exception {
        Path modelPath = Path.of(System.getProperty("probe.model"));
        int requests = Integer.getInteger("probe.requests", 64);
        int batch = Integer.getInteger("probe.b", 16);
        int contextLength = Integer.getInteger("probe.ctx", 512);
        int maxNewTokens = Integer.getInteger("probe.n", 64);
        String prompt = System.getProperty("probe.prompt", "What is the capital of France?");
        System.setProperty("jitllm.prefillBatchSize", String.valueOf(batch));

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
            try (LLMEngine engine = new LLMEngine(model, manager, executor, batch, requests * 2)) {
                int[] prepared = prepare(model, prompt);

                // Warm-up outside the timing, as the harness excludes its first steps.
                drive(engine, submit(engine, prepared, batch, maxNewTokens));

                long start = System.nanoTime();
                List<RequestHandle> handles = submit(engine, prepared, requests, maxNewTokens);
                int steps = drive(engine, handles);
                double seconds = (System.nanoTime() - start) / 1e9;

                int generated = 0;
                for (RequestHandle handle : handles) {
                    generated += handle.tokenCount();
                }
                System.out.printf(
                        "[ENGINE-129] B=%d requests=%d steps=%d generated=%d "
                                + "wall=%.3fs aggregate=%.1f tok/s requests/s=%.1f "
                                + "prefillTokens=%d%n",
                        batch,
                        requests,
                        steps,
                        generated,
                        seconds,
                        generated / seconds,
                        requests / seconds,
                        (long) requests * prepared.length);
                System.out.println(
                        "[ENGINE-129] first answer: "
                                + model.tokenizer()
                                        .decode(handles.get(0).tokens())
                                        .replace("\n", " "));
            }
        }
        manager.close();
    }

    private static int[] prepare(Model model, String text) {
        ChatFormat chatFormat = model.chatFormat();
        List<Integer> tokens = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            tokens.add(chatFormat.getBeginOfText());
        }
        tokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, text)));
        tokens.addAll(
                chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        int[] prepared = new int[tokens.size()];
        for (int i = 0; i < prepared.length; i++) {
            prepared[i] = tokens.get(i);
        }
        return prepared;
    }

    private static List<RequestHandle> submit(
            LLMEngine engine, int[] prompt, int count, int budget) {
        List<RequestHandle> handles = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            handles.add(engine.addRequest(prompt, budget, null));
        }
        return handles;
    }

    private static int drive(LLMEngine engine, List<RequestHandle> handles) {
        int steps = 0;
        while (steps < 100_000) {
            boolean done = true;
            for (RequestHandle handle : handles) {
                if (!handle.isTerminal()) {
                    done = false;
                    break;
                }
            }
            if (done) {
                return steps;
            }
            engine.step();
            steps++;
        }
        return steps;
    }
}
