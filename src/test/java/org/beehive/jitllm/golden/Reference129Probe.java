package org.beehive.jitllm.golden;

import java.nio.file.Path;
import org.beehive.jitllm.backend.tornado.bench.BatchDecodeOptions;
import org.beehive.jitllm.backend.tornado.bench.BatchedDecodeEngine;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.ModelLoader;

/**
 * Drives #129's own harness, so its numbers can be compared with the promoted path's.
 *
 * <p>{@code BatchedDecodeEngine} exposes {@code run(model, prompt)} and no {@code main} — its class
 * javadoc still describes swapping a main class that is not there, which is why nobody had run it
 * since it was absorbed. This is the missing driver, in test scope: the bench is frozen and gains
 * nothing production-shaped from being made executable.
 *
 * <p>{@code -Dprobe.model=… -Dprobe.batch=16 -Djitllm.prefillBatchSize=16 …}
 */
public final class Reference129Probe {

    public static void main(String[] args) throws Exception {
        Path modelPath = Path.of(System.getProperty("probe.model"));
        String prompt = System.getProperty("probe.prompt", "What is the capital of France?");
        int batchSize = Integer.getInteger("probe.batch", 32);
        int contextLength = Integer.getInteger("probe.ctx", 512);

        Model model = ModelLoader.loadModel(modelPath, contextLength, true, true);
        var options =
                new BatchDecodeOptions(
                        batchSize,
                        contextLength,
                        64,
                        true,
                        false,
                        16,
                        0,
                        false,
                        4 * batchSize,
                        32,
                        false,
                        0.0f,
                        true);
        BatchedDecodeEngine.Report report = BatchedDecodeEngine.run(model, prompt, options);
        System.out.println("[REF129] " + options);
        System.out.println("[REF129] " + report);
    }
}
