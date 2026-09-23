package org.beehive.jitllm.integration.cli;

import java.util.Locale;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;
import org.beehive.jitllm.api.ModelConfiguration;
import org.beehive.jitllm.api.ModelInfo;
import org.beehive.jitllm.auxiliary.metrics.RunMetricsSnapshot;
import org.beehive.jitllm.format.GgufModelFacts;
import org.beehive.jitllm.runtime.backend.ExecutionInfo;
import org.beehive.jitllm.runtime.memory.MemoryPlan;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/** CLI-only, aligned startup diagnostics inspired by llama.cpp's model information rows. */
record StartupSummary(
        ModelInfo model,
        ModelConfiguration shape,
        ExecutionInfo execution,
        GgufModelFacts file,
        long fileBytes,
        String sampling,
        long modelLoadNs,
        long readyNs,
        RunMetricsSnapshot timings,
        MemoryPlan memory,
        int concurrentSessions,
        boolean verbose) {

    String render() {
        StringBuilder out =
                new StringBuilder("\n── jitllm · run configuration ────────────────────────────\n");
        row(out, "Device", execution.backend() + " / " + execution.device());
        boolean gpu = !execution.backend().equals("CPU");
        if (gpu) {
            row(out, "Runtime", execution.runtime());
        }
        row(out, "Model", model.source() == null ? model.name() : model.source().getFileName());
        row(
                out,
                "Architecture",
                (file == null ? model.architecture() : file.arch())
                        + " / "
                        + shape.layers()
                        + " layers");
        if (verbose) {
            row(
                    out,
                    "Dimensions / heads",
                    shape.dimension()
                            + " embedding / "
                            + shape.attentionHeads()
                            + " attention heads / "
                            + shape.keyValueHeads()
                            + " KV heads");
            trainingContext(shape)
                    .ifPresent(tokens -> row(out, "Training context", tokens + " tokens"));
            if (gpu && timings.executionPath() != null) {
                row(out, "Execution path", timings.executionPath());
            }
        }
        if (file != null) {
            row(
                    out,
                    "Model size",
                    String.format(Locale.ROOT, "%.3f B parameters", file.paramsB())
                            + (fileBytes >= 0
                                    ? String.format(
                                            Locale.ROOT,
                                            " / %.2f GiB file",
                                            fileBytes / 1073741824.0)
                                    : ""));
        }
        if (gpu) {
            row(
                    out,
                    "GPU memory budget",
                    System.getProperty("tornado.device.memory", "runtime default")
                            + " (allocation limit)");
        }
        if (gpu) {
            printMemory(out);
        }
        row(
                out,
                "Quantization",
                "GGUF "
                        + (file == null ? "unknown" : file.quant())
                        + " / loaded "
                        + model.weightTypes().stream()
                                .map(Enum::name)
                                .sorted()
                                .collect(Collectors.joining(", ")));
        row(out, "Execution", execution.mode());
        row(out, "Context", model.contextLength() + " tokens");
        if (concurrentSessions > 0) {
            row(out, "Concurrent sessions", concurrentSessions);
        }
        if (execution.prefillBatchSize() > 1) {
            row(out, "Prefill chunk", execution.prefillBatchSize() + " tokens");
        }
        row(out, "KV cache", execution.kvCache());
        if (gpu) {
            row(
                    out,
                    "MMA / tensor cores",
                    execution.prefillKernels()
                            + (execution.mode().equals("continuous-batch-decode")
                                    ? " (batched prefill/decode)"
                                    : " (prefill)"));
            row(out, "Native libraries", execution.nativeLibraries());
            row(out, "CUDA graphs", execution.cudaGraphs() ? "on" : "off");
            row(out, "Staged transfers", execution.stagedTransfers() ? "on" : "off");
        } else {
            row(
                    out,
                    "CPU threads",
                    ForkJoinPool.getCommonPoolParallelism() + " common-pool workers + caller");
            int vectorBits = FloatTensor.vectorBitSize();
            row(
                    out,
                    "Vector API",
                    (vectorBits == 0 ? "tensor SIMD off" : "tensor SIMD " + vectorBits + "-bit")
                            + (model.weightTypes()
                                                    .contains(
                                                            org.beehive.jitllm.runtime.tensor.DataType
                                                                    .Q4_0)
                                            || model.weightTypes()
                                                    .contains(
                                                            org.beehive.jitllm.runtime.tensor.DataType
                                                                    .Q4_1)
                                    ? "; Q4 SIMD "
                                            + (Boolean.parseBoolean(
                                                            System.getProperty(
                                                                    "jitllm.VectorAPI", "true"))
                                                    ? "on"
                                                    : "off")
                                    : ""));
        }
        row(out, "Sampling", sampling);
        out.append("\n  Initialization\n");
        row(out, "Model load", milliseconds(modelLoadNs));
        if (gpu) {
            row(out, "Plan construction", setupTime(timings.tornadoPlanCreationDuration()));
            row(out, "JIT precompilation", setupTime(timings.tornadoJitDuration()));
            row(
                    out,
                    "Initial device setup",
                    setupTime(timings.tornadoReadOnlyWeightsCopyInDuration())
                            + " (uploads + execution / graph capture)");
        }
        row(out, "Ready to generate", milliseconds(readyNs));
        out.append("──────────────────────────────────────────────────────\n\n");
        return out.toString();
    }

    private void printMemory(StringBuilder out) {
        if (memory == null || memory.confidence() == MemoryPlan.Confidence.UNSUPPORTED) {
            row(out, "GPU memory estimate", "unavailable");
            return;
        }
        long weights = 0, kv = 0, workspace = 0;
        for (var component : memory.components()) {
            switch (component.bufferClass()) {
                case WEIGHTS_PER_LAYER, WEIGHTS_GLOBAL -> weights += component.predictedBytes();
                case KV_CACHE -> kv += component.predictedBytes();
                default -> workspace += component.predictedBytes();
            }
        }
        row(
                out,
                "GPU memory estimate",
                String.format(
                                Locale.ROOT,
                                "%.2f GiB total / %.2f MiB weights / %.2f MiB KV / %.2f MiB workspace",
                                memory.predictedBudgetBytes() / 1073741824.0,
                                weights / 1048576.0,
                                kv / 1048576.0,
                                workspace / 1048576.0)
                        + (memory.confidence() == MemoryPlan.Confidence.CONSERVATIVE
                                ? " (conservative)"
                                : ""));
        if (verbose) {
            row(out, "Estimate assumptions", memory.confidence() + " / " + memory.assumptions());
        }
    }

    private static String setupTime(long ns) {
        return ns > 0 ? milliseconds(ns) : "not measured (lazy execution)";
    }

    private static String milliseconds(long ns) {
        return String.format(Locale.ROOT, "%.2f ms", ns / 1e6);
    }

    /**
     * The model's own maximum sequence length, when its family records one. Llama, Mistral and
     * Devstral configurations do not, and throw; a diagnostic report must not fail the run for it.
     */
    private static java.util.OptionalInt trainingContext(ModelConfiguration shape) {
        try {
            return java.util.OptionalInt.of(shape.maxContextLength());
        } catch (UnsupportedOperationException notRecorded) {
            return java.util.OptionalInt.empty();
        }
    }

    private static void row(StringBuilder out, String label, Object value) {
        // Keep metadata on one line, including filenames containing control characters.
        out.append(
                String.format(
                        Locale.ROOT,
                        "  %-21s : %s%n",
                        label,
                        value.toString().replaceAll("\\p{Cntrl}", " ")));
    }
}
