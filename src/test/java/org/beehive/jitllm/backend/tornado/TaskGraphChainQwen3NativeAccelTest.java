package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.Map;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

/**
 * {@code --print-taskgraph-chain} on the Qwen3 0.6B F16 batched plan with native libraries: the
 * cuBLAS and cuDNN library tasks appear with their functions and named arguments, both prefill
 * families and the grouped decode graphs are labelled, and the schedule says which family runs
 * when.
 */
public class TaskGraphChainQwen3NativeAccelTest {

    @Test
    public void theBatchedNativePlanPrintsLibraryTasksFamiliesAndGrouping() throws Exception {
        Path file = GoldenFixture.locate(Fixture.QWEN3_0_6B_F16);
        assumeTrue(
                "environment absent: " + GoldenFixture.absentMessage(Fixture.QWEN3_0_6B_F16),
                file != null);
        assumeTrue(
                "the native path needs tensor cores",
                TensorCoreSupport.isTensorCoreCapableBackend());
        assumeTrue("cuBLAS is not loadable here", NativePrefillSupport.cublasAvailable());
        String chain =
                TaskGraphChainLlamaAccelTest.capture(
                        file,
                        Map.of(
                                "jitllm.withPrefillDecode",
                                "true",
                                "jitllm.prefillBatchSize",
                                "128",
                                ExecutionPolicy.NATIVE_LIBRARIES_PROPERTY,
                                "true"));

        assertTrue(
                chain,
                chain.startsWith(
                        "Plan  QWEN_3 F16 · batch-prefill-decode (batch 128) · 20 graphs"));
        // the schedule names both prefill families and when each runs
        assertTrue(chain, chain.contains("prefill activation → 7 × prefill layers "));
        assertTrue(
                chain,
                chain.contains(
                        "prefill activation → 7 × prefill fallback layers  per later chunk"));
        assertTrue(chain, chain.contains("decode activation → 3 × decode layers → logits"));
        // grouped graphs: one block per family, layers per graph
        assertTrue(chain, chain.contains("(prefill layers, 7 graphs, 4 layers each)"));
        assertTrue(chain, chain.contains("(prefill fallback layers, 7 graphs, 4 layers each)"));
        assertTrue(chain, chain.contains("(decode layers, 3 graphs, 10/10/8 layers)"));
        assertTrue(chain, chain.contains("× 4 layers per graph (L0_ … L3_)"));
        // library calls with what they compute
        assertTrue(
                chain,
                chain.matches(
                        "(?s).*libraryTask +qkvProj +cuBLAS cublasGemmExFP16FP32 +m=4096 n=128"
                                + " k=1024\n.*"));
        assertTrue(
                chain,
                chain.matches(
                        "(?s).*libraryTask +cudnn_sdpa +cuDNN sdpaForward +b=1 h=16 q=128 kv=128"
                                + " d=128 causal\n.*"));
        assertTrue(chain, chain.contains("task         cudnn_pack_q"));
        assertFalse(chain, chain.contains("(unavailable"));
        assertTrue(
                "a summary, not a dump: " + chain.lines().count() + " lines",
                chain.lines().count() < 80);
    }
}
