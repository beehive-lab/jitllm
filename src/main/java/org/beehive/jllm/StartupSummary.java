package org.beehive.jllm;

import java.util.Locale;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;
import org.beehive.jllm.api.ModelConfiguration;
import org.beehive.jllm.api.ModelInfo;
import org.beehive.jllm.auxiliary.metrics.RunMetricsSnapshot;
import org.beehive.jllm.format.GgufModelFacts;
import org.beehive.jllm.runtime.backend.ExecutionInfo;
import org.beehive.jllm.runtime.memory.MemoryPlan;
import org.beehive.jllm.tensor.standard.FloatTensor;

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
        boolean verbose) {

    String render() {
        StringBuilder out =
                new StringBuilder("\n── jllm · run configuration ────────────────────────────\n");
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
            row(out, "Training context", shape.maxContextLength() + " tokens");
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
        if (execution.prefillBatchSize() > 1) {
            row(out, "Prefill chunk", execution.prefillBatchSize() + " tokens");
        }
        row(out, "KV cache", execution.kvCache());
        if (gpu) {
            row(out, "MMA / tensor cores", execution.prefillKernels() + " (prefill)");
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
                                                            org.beehive.jllm.runtime.tensor.DataType
                                                                    .Q4_0)
                                            || model.weightTypes()
                                                    .contains(
                                                            org.beehive.jllm.runtime.tensor.DataType
                                                                    .Q4_1)
                                    ? "; Q4 SIMD "
                                            + (Boolean.parseBoolean(
                                                            System.getProperty(
                                                                    "jllm.VectorAPI", "true"))
                                                    ? "on"
                                                    : "off")
                                    : ""));
        }
        row(out, "Sampling", sampling);
        out.append("\n  Initialization\n");
        row(out, "Model load", milliseconds(modelLoadNs));
        if (gpu) {
            row(out, "Plan construction", milliseconds(timings.tornadoPlanCreationDuration()));
            row(out, "JIT precompilation", milliseconds(timings.tornadoJitDuration()));
            row(
                    out,
                    "Initial device setup",
                    milliseconds(timings.tornadoReadOnlyWeightsCopyInDuration())
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

    private static String milliseconds(long ns) {
        return String.format(Locale.ROOT, "%.2f ms", ns / 1e6);
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
