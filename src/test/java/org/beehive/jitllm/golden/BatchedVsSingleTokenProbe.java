package org.beehive.jitllm.golden;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.backend.tornado.batch.TornadoBatchExecutor;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.runtime.batch.BatchSlots;
import org.beehive.jitllm.runtime.kv.KvCacheManager;
import org.beehive.jitllm.runtime.kv.KvLease;
import org.beehive.jitllm.runtime.kv.KvStorage;
import org.beehive.jitllm.runtime.kv.KvStorageFactories;
import org.beehive.jitllm.runtime.kv.KvStorageRequest;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Does the batched MMA decode path produce the same greedy tokens as single-token decode?
 *
 * <p>{@code -Dprobe.model=… -Dprobe.tokens=32}
 */
public final class BatchedVsSingleTokenProbe {

    public static void main(String[] args) throws Exception {
        Path modelPath = Path.of(System.getProperty("probe.model"));
        int howMany = Integer.getInteger("probe.tokens", 32);
        int contextLength = 512;
        int batch = 2;

        List<Integer> prompt;
        List<Integer> singleTokenOutput;

        // ── single-token path, greedy ────────────────────────────────────────────────────────
        {
            Model model = ModelLoader.loadModel(modelPath, contextLength, true, true);
            ChatFormat cf = model.chatFormat();
            prompt = new ArrayList<>();
            if (model.shouldAddBeginOfText()) {
                prompt.add(cf.getBeginOfText());
            }
            prompt.addAll(
                    cf.encodeMessage(
                            new ChatFormat.Message(
                                    ChatFormat.Role.USER,
                                    System.getProperty("probe.prompt", "Name three colours."))));
            prompt.addAll(cf.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

            State state = model.createNewState();
            TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
            singleTokenOutput = new ArrayList<>();
            try {
                int token = prompt.get(0);
                for (int position = 0; position < prompt.size() + howMany; position++) {
                    org.beehive.jitllm.inference.Logits logits =
                            org.beehive.jitllm.backend.tornado.TornadoForwardPass.forward(
                                    model, state, token, position, plan);
                    int next = argmax(logits);
                    if (position + 1 < prompt.size()) {
                        token = prompt.get(position + 1);
                    } else {
                        token = next;
                        singleTokenOutput.add(next);
                    }
                }
            } finally {
                plan.freeTornadoExecutionPlan();
            }
        }

        // ── batched path, greedy, one active slot ────────────────────────────────────────────
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
        KvLease lease = manager.acquire(contextLength);
        State state = model.createNewState(lease);

        List<Integer> batchedOutput = new ArrayList<>();
        try (TornadoBatchExecutor executor =
                new TornadoBatchExecutor(model, state, store, batch, blocksPerSlot)) {
            boolean[] active = {true, false};
            int[] tokens = new int[batch];
            int[] positions = new int[batch];
            int[] kvSlots = {0, 1};

            int token = prompt.get(0);
            for (int position = 0; position < prompt.size() + howMany; position++) {
                tokens[0] = token;
                positions[0] = position;
                int[] sampled =
                        executor.decodeStep(new BatchSlots(active, tokens, positions, kvSlots));
                int next = sampled[0];
                if (position + 1 < prompt.size()) {
                    token = prompt.get(position + 1);
                } else {
                    token = next;
                    batchedOutput.add(next);
                }
            }
        }
        lease.close();
        manager.close();

        System.out.println("[PROBE] prompt tokens: " + prompt.size());
        System.out.println("[PROBE] single-token: " + singleTokenOutput);
        System.out.println("[PROBE] batched     : " + batchedOutput);
        int agree = 0;
        for (int i = 0; i < Math.min(singleTokenOutput.size(), batchedOutput.size()); i++) {
            if (singleTokenOutput.get(i).equals(batchedOutput.get(i))) {
                agree++;
            } else {
                break;
            }
        }
        System.out.println("[PROBE] identical prefix: " + agree + "/" + singleTokenOutput.size());
        System.out.println("[PROBE] byte-identical: " + singleTokenOutput.equals(batchedOutput));
    }

    private static int argmax(org.beehive.jitllm.inference.Logits logits) {
        int best = 0;
        float bestValue = logits.get(0);
        for (int i = 1; i < logits.size(); i++) {
            if (logits.get(i) > bestValue) {
                bestValue = logits.get(i);
                best = i;
            }
        }
        return best;
    }

    private static int argmax(FloatArray logits) {
        int best = 0;
        float bestValue = logits.get(0);
        for (int i = 1; i < logits.getSize(); i++) {
            if (logits.get(i) > bestValue) {
                bestValue = logits.get(i);
                best = i;
            }
        }
        return best;
    }
}
