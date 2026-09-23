package org.beehive.jitllm.integration.cli;

import static org.junit.Assert.*;

import java.nio.file.Path;
import java.util.Locale;
import java.util.Set;
import org.beehive.jitllm.api.ModelConfiguration;
import org.beehive.jitllm.api.ModelInfo;
import org.beehive.jitllm.auxiliary.metrics.RunMetricsSnapshot;
import org.beehive.jitllm.format.GgufModelFacts;
import org.beehive.jitllm.runtime.backend.ExecutionInfo;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

public class StartupSummaryTest {
    private static final ModelConfiguration SHAPE =
            new ModelConfiguration() {
                public int dimension() {
                    return 1024;
                }

                public int hiddenDimension() {
                    return 3072;
                }

                public int layers() {
                    return 28;
                }

                public int attentionHeads() {
                    return 16;
                }

                public int keyValueHeads() {
                    return 8;
                }

                public int vocabularySize() {
                    return 151936;
                }

                public int maxContextLength() {
                    return 32768;
                }
            };

    private StartupSummary summary(ExecutionInfo execution) {
        return new StartupSummary(
                new ModelInfo(
                        "QWEN_3",
                        "qwen_3",
                        2048,
                        Path.of("Qwen3-0.6B-Q4_0.gguf"),
                        Set.of(DataType.Q8_0, DataType.F32),
                        DataType.Q8_0),
                SHAPE,
                execution,
                new GgufModelFacts("qwen3", "Q4_0", 0.6),
                536870912,
                "greedy",
                1_500_000,
                10_000_000,
                RunMetricsSnapshot.of(
                        0, 0, 0, 0, 0, 0, false, 2_000_000, 3_000_000, 4_000_000, null, null, null,
                        null),
                null,
                1,
                false);
    }

    @Test
    public void distinguishesFileQuantizationFromLoadedWeightsAndLabelsSetupHonestly() {
        Locale old = Locale.getDefault();
        try {
            Locale.setDefault(Locale.GERMANY);
            String text =
                    summary(
                                    new ExecutionInfo(
                                            "CUDA",
                                            "test GPU",
                                            "TornadoVM test",
                                            "batch-prefill-decode",
                                            32,
                                            "FP16",
                                            "FP16 tensor-core MMA",
                                            "JIT kernels",
                                            true,
                                            true))
                            .render();
            assertTrue(text.contains("Qwen3-0.6B-Q4_0.gguf"));
            assertTrue(text.contains("0.600 B parameters / 0.50 GiB file"));
            assertTrue(text.contains("Quantization          : GGUF Q4_0 / loaded F32, Q8_0"));
            assertTrue(text.indexOf("Device") < text.indexOf("Model"));
            assertTrue(text.indexOf("Model size") < text.indexOf("GPU memory budget"));
            assertTrue(text.indexOf("GPU memory budget") < text.indexOf("Quantization"));
            assertTrue(text.contains("Runtime               : TornadoVM test"));
            assertFalse(text.contains("Prefill projections"));
            assertFalse(text.contains("Prefill attention"));
            assertTrue(text.contains("MMA / tensor cores    : FP16 tensor-core MMA (prefill)"));
            assertTrue(text.contains("KV cache              : FP16"));
            assertTrue(text.contains("Context               : 2048 tokens"));
            assertTrue(text.contains("Prefill chunk         : 32 tokens"));
            assertTrue(text.contains("JIT precompilation    : 3.00 ms"));
            assertTrue(text.contains("4.00 ms (uploads + execution / graph capture)"));
            assertTrue(text.contains("Ready to generate     : 10.00 ms"));
        } finally {
            Locale.setDefault(old);
        }
    }

    @Test
    public void cpuDoesNotClaimGpuTimingsOrMemory() {
        String text =
                summary(
                                new ExecutionInfo(
                                        "CPU",
                                        "test CPU",
                                        "Java test",
                                        "single-token",
                                        1,
                                        "FP32",
                                        "CPU kernels",
                                        "CPU kernels",
                                        false,
                                        false))
                        .render();
        assertFalse(text.contains("Runtime               :"));
        assertFalse(text.contains("CUDA graphs"));
        assertFalse(text.contains("Staged transfers"));
        assertFalse(text.contains("MMA / tensor cores"));
        assertFalse(text.contains("Prefill chunk"));
        assertTrue(text.contains("CPU threads"));
        assertTrue(text.contains("Vector API"));
        assertFalse(text.contains("JIT precompilation"));
        assertFalse(text.contains("GPU memory budget"));
        assertTrue(text.contains("Model load            : 1.50 ms"));
    }

    @Test
    public void estimatesIncludeAllBufferClassesAndVerboseDetailsAreOptional() {
        var base =
                summary(
                        new ExecutionInfo(
                                "CUDA",
                                "GPU",
                                "runtime",
                                "batch-prefill-decode",
                                32,
                                "FP16",
                                "library-selected",
                                "native",
                                true,
                                true,
                                "cuBLAS, cuDNN"));
        long gib = 1073741824L;
        var memory =
                new org.beehive.jitllm.runtime.memory.MemoryPlan(
                        java.util.List.of(
                                new org.beehive.jitllm.runtime.memory.MemoryComponent(
                                        "weights",
                                        org.beehive.jitllm.runtime.memory.BufferClass
                                                .WEIGHTS_PER_LAYER,
                                        gib,
                                        2,
                                        0),
                                new org.beehive.jitllm.runtime.memory.MemoryComponent(
                                        "cache",
                                        org.beehive.jitllm.runtime.memory.BufferClass.KV_CACHE,
                                        gib,
                                        1,
                                        0),
                                new org.beehive.jitllm.runtime.memory.MemoryComponent(
                                        "scratch",
                                        org.beehive.jitllm.runtime.memory.BufferClass.BATCH_STAGING,
                                        gib,
                                        1,
                                        0),
                                new org.beehive.jitllm.runtime.memory.MemoryComponent(
                                        "controls",
                                        org.beehive.jitllm.runtime.memory.BufferClass.CONTROL,
                                        gib,
                                        1,
                                        0)),
                        10 * gib,
                        org.beehive.jitllm.runtime.memory.MemoryPlan.Confidence.CONSERVATIVE,
                        "test topology");
        for (boolean verbose : new boolean[] {false, true}) {
            var report =
                    new StartupSummary(
                            base.model(),
                            base.shape(),
                            base.execution(),
                            base.file(),
                            base.fileBytes(),
                            base.sampling(),
                            base.modelLoadNs(),
                            base.readyNs(),
                            base.timings(),
                            memory,
                            base.concurrentSessions(),
                            verbose);
            String text = report.render();
            assertTrue(
                    text.contains(
                            "5.00 GiB total / 2048.00 MiB weights / 1024.00 MiB KV / 2048.00 MiB workspace (conservative)"));
            assertTrue(text.contains("Native libraries      : cuBLAS, cuDNN"));
            assertEquals(verbose, text.contains("Dimensions / heads"));
            assertEquals(verbose, text.contains("Training context"));
            assertEquals(verbose, text.contains("test topology"));
        }
    }

    @Test
    public void finalPerformanceBlockDoesNotRepeatStartupEvenInVerboseMode() {
        String old = System.getProperty("jitllm.EnableTimingForTornadoVMInit");
        try {
            System.setProperty("jitllm.EnableTimingForTornadoVMInit", "true");
            String text =
                    new org.beehive.jitllm.auxiliary.metrics.HumanMetricsRenderer()
                            .render(
                                    summary(
                                                    new ExecutionInfo(
                                                            "CPU",
                                                            "CPU",
                                                            "Java",
                                                            "single-token",
                                                            1,
                                                            "FP32",
                                                            "",
                                                            "",
                                                            false,
                                                            false))
                                            .timings());
            assertFalse(text.contains("Plan construction"));
            assertFalse(text.contains("Model Load"));
            assertFalse(text.contains("JIT precompilation"));
        } finally {
            if (old == null) System.clearProperty("jitllm.EnableTimingForTornadoVMInit");
            else System.setProperty("jitllm.EnableTimingForTornadoVMInit", old);
        }
    }
}
