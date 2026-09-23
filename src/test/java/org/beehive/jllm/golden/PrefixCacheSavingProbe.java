package org.beehive.jllm.golden;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.beehive.jllm.backend.tornado.batch.TornadoBatchExecutor;
import org.beehive.jllm.engine.LLMEngine;
import org.beehive.jllm.engine.RequestHandle;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.format.ChatFormat;
import org.beehive.jllm.model.loader.ModelLoader;
import org.beehive.jllm.runtime.kv.KvCacheManager;
import org.beehive.jllm.runtime.kv.KvLease;
import org.beehive.jllm.runtime.kv.KvStorage;
import org.beehive.jllm.runtime.kv.KvStorageFactories;
import org.beehive.jllm.runtime.kv.KvStorageRequest;

/**
 * What a prefix cache saves on a device, and whether the answers stay the same.
 *
 * <p>#129 demonstrated prefix savings in its own harness. This measures the promoted path: a shared
 * opening served repeatedly, with the cache off and then on, comparing both the time to first token
 * and — the part that matters more — the tokens produced.
 *
 * <p>{@code -Dprobe.model=… -Dprobe.requests=8}
 */
public final class PrefixCacheSavingProbe {

    public static void main(String[] args) throws Exception {
        Path modelPath = Path.of(System.getProperty("probe.model"));
        int requests = Integer.getInteger("probe.requests", 8);
        int batch = Integer.getInteger("probe.b", 4);
        int contextLength = 512;
        System.setProperty("jllm.prefillBatchSize", String.valueOf(batch));

        // A long shared opening, as served traffic has: the same system framing every time.
        String shared =
                "You are a careful assistant. Answer in one short sentence. "
                        + "Be precise, be brief, and never speculate beyond what is asked. ";

        List<Integer> withoutCache = run(modelPath, batch, contextLength, requests, shared, 0);
        List<Integer> withCache = run(modelPath, batch, contextLength, requests, shared, 16);

        System.out.println("[PREFIX] tokens identical: " + withoutCache.equals(withCache));
        if (!withoutCache.equals(withCache)) {
            System.out.println("[PREFIX]   without: " + withoutCache);
            System.out.println("[PREFIX]   with   : " + withCache);
        }
    }

    private static List<Integer> run(
            Path modelPath,
            int batch,
            int contextLength,
            int requests,
            String shared,
            int prefixEntries)
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
        if (prefixEntries > 0) {
            manager.enablePrefixCache(prefixEntries);
        }

        KvLease planLease = manager.acquire(blockTokens);
        State state = model.createNewState(planLease);
        List<Integer> firstAnswer = new ArrayList<>();

        try (TornadoBatchExecutor executor =
                new TornadoBatchExecutor(model, state, store, batch, blocksPerSlot)) {
            planLease.close();
            try (LLMEngine engine = new LLMEngine(model, manager, executor, batch, batch * 4)) {
                ChatFormat chatFormat = model.chatFormat();
                int[] prompt = prepare(model, chatFormat, shared + "Name one colour.");

                // Warm-up, so compilation is not in the timing.
                drive(engine, List.of(engine.addRequest(prompt, 4, null)));

                long start = System.nanoTime();
                List<RequestHandle> handles = new ArrayList<>();
                for (int i = 0; i < requests; i++) {
                    handles.add(engine.addRequest(prompt, 12, null));
                }
                drive(engine, handles);
                long elapsed = System.nanoTime() - start;

                long ttftSum = 0;
                for (RequestHandle handle : handles) {
                    ttftSum += handle.timeToFirstTokenNanos();
                }
                firstAnswer.addAll(handles.get(0).tokens());

                System.out.printf(
                        "[PREFIX] cache=%-3s  %d requests in %.2fs  mean TTFT %.0f ms"
                                + "  blocks reused %d%n",
                        prefixEntries > 0 ? "on" : "off",
                        requests,
                        elapsed / 1e9,
                        ttftSum / 1e6 / requests,
                        manager.prefixCache() == null ? 0 : manager.prefixCache().blocksReused());
            }
        }
        manager.close();
        return firstAnswer;
    }

    private static int[] prepare(Model model, ChatFormat chatFormat, String text) {
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

    private static void drive(LLMEngine engine, List<RequestHandle> handles) {
        int guard = 0;
        while (guard++ < 8192) {
            boolean done = true;
            for (RequestHandle handle : handles) {
                if (!handle.isTerminal()) {
                    done = false;
                    break;
                }
            }
            if (done) {
                return;
            }
            engine.step();
        }
    }
}
