package org.beehive.jitllm.integration.cli;

import java.io.IOException;
import java.nio.file.Files;
import org.beehive.jitllm.api.LocalModel;
import org.beehive.jitllm.api.LocalModels;
import org.beehive.jitllm.api.ModelConfiguration;
import org.beehive.jitllm.api.ModelInfo;
import org.beehive.jitllm.api.ModelOptions;
import org.beehive.jitllm.auxiliary.RunMetrics;
import org.beehive.jitllm.format.GgufModelFacts;
import org.beehive.jitllm.runtime.backend.ExecutionInfo;
import org.beehive.jitllm.runtime.memory.MemoryPlan;

/** Shared startup-report assembly; the integration chooses when and where to print it. */
public final class StartupDiagnostics {
    private StartupDiagnostics() {}

    /**
     * Prints {@code --print-taskgraph-chain} output as plain text on stderr. The backend only
     * renders it; where it goes is the integration's choice.
     */
    public static void installTaskGraphChainOutput() {
        org.beehive.jitllm.backend.tornado.TaskGraphChainPrinter.output(
                text -> {
                    System.err.print(text);
                    System.err.flush();
                });
    }

    public static boolean verbose() {
        return Boolean.getBoolean("jitllm.verbose")
                || Boolean.getBoolean("jitllm.EnableTimingForTornadoVMInit");
    }

    public static String render(
            org.beehive.jitllm.model.Model model,
            java.nio.file.Path path,
            ExecutionInfo execution,
            String sampling,
            ModelOptions options,
            long loadNs,
            long startedNs) {
        var c = model.configuration();
        ModelInfo info =
                new ModelInfo(
                        path.getFileName().toString(),
                        model.architectureId().toString(),
                        c.contextLength(),
                        path,
                        java.util.Set.of(model.weights().dataType()),
                        c.activationType());
        ModelConfiguration shape =
                new ModelConfiguration() {
                    public int dimension() {
                        return c.dim();
                    }

                    public int hiddenDimension() {
                        return c.hiddenDim();
                    }

                    public int layers() {
                        return c.numberOfLayers();
                    }

                    public int attentionHeads() {
                        return c.numberOfHeads();
                    }

                    public int keyValueHeads() {
                        return c.numberOfKeyValueHeads();
                    }

                    public int vocabularySize() {
                        return c.vocabularySize();
                    }

                    public int maxContextLength() {
                        return c.contextLengthModel();
                    }
                };
        // A raw model has no facade sessions: its caller drives one plan, or the batch engine.
        return render(info, shape, execution, sampling, options, 0, loadNs, startedNs);
    }

    public static String render(
            LocalModel model,
            ExecutionInfo execution,
            String sampling,
            ModelOptions modelOptions,
            long modelLoadNs,
            long startedNs) {
        return render(
                model.info(),
                model.configuration(),
                execution,
                sampling,
                modelOptions,
                modelLoadNs,
                startedNs);
    }

    public static String render(
            ModelInfo info,
            ModelConfiguration shape,
            ExecutionInfo execution,
            String sampling,
            ModelOptions modelOptions,
            long modelLoadNs,
            long startedNs) {
        return render(
                info,
                shape,
                execution,
                sampling,
                modelOptions,
                modelOptions.maxConcurrentSessions(),
                modelLoadNs,
                startedNs);
    }

    private static String render(
            ModelInfo info,
            ModelConfiguration shape,
            ExecutionInfo execution,
            String sampling,
            ModelOptions modelOptions,
            int concurrentSessions,
            long modelLoadNs,
            long startedNs) {
        GgufModelFacts facts = null;
        long bytes = -1;
        try {
            if (info.source() != null) {
                facts = GgufModelFacts.read(info.source());
                bytes = Files.size(info.source());
            }
        } catch (IOException e) {
            // Optional file diagnostics must not prevent an already loaded model from running.
        }
        var initialMetrics = RunMetrics.snapshot();
        MemoryPlan memory = null;
        if (!execution.backend().equals("CPU")
                && !execution.mode().equals("continuous-batch-decode")) {
            try {
                memory = LocalModels.preflight(info.source(), modelOptions);
            } catch (IOException | RuntimeException e) {
                // Diagnostics are optional; preserve the successfully prepared session.
            } finally {
                // Metadata-only preflight uses the loader, which also records a load duration.
                RunMetrics.setLoadDuration(initialMetrics.loadDuration());
            }
        }
        return new StartupSummary(
                        info,
                        shape,
                        execution,
                        facts,
                        bytes,
                        sampling,
                        modelLoadNs,
                        System.nanoTime() - startedNs,
                        initialMetrics,
                        memory,
                        concurrentSessions,
                        true)
                .render();
    }
}
