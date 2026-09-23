package org.beehive.jllm.golden;

import org.beehive.jllm.backend.tornado.PlanDispatchEvidence;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

// @formatter:off
/**
 * The batched prefill parity at width 256 with the tensor-core path selected — the width at which
 * the Q4_0 projections wide enough for it run as a dequantization into FP16 scratch followed by the
 * tiled FP16 GEMM. The 63-token fixture is one partial chunk of the 256-row launch, so the pair's
 * inactive rows are computed and never read. The dispatch is asserted off this run's own plan: the
 * attention-output projection's dequantization task and GEMM grid in every attention layer.
 */
// @formatter:on
public class Qwen35MmaBatchedPrefillWidth256ParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jllm.golden.Fp32KeyValueCache();

    static {
        System.setProperty("jllm.qwen35.tensorCores", "true");
    }

    @Test
    public void qwen3_8_27b_q4_0_batchedPrefillParityAt256OnTensorCores() throws Exception {
        GoldenCapture.Result gpu =
                assertParityBatched(Fixture.QWEN3_8_27B_Q4_0, Q8_0_PACKED_DECODE, 256);
        // The FP32 cache: attention is the FP32-cache kernel here, not the scored family the
        // benchmarked FP16 cache selects (Qwen35MmaBatchedPrefillWidth256Fp16KvParityAccelTest).
        org.junit.Assert.assertEquals(
                "the batched attention kernel this plan compiled",
                java.util.Set.of("attentionBatchPaged"),
                gpu.batchedTaskKernels.get("attention"));
        PlanDispatchEvidence.assertQwen35AttentionOutputOnDequantGemm(
                gpu.gridScheduler, 256, gpu.dim, 6144);
        PlanDispatchEvidence.assertQwen35SsmOutOnDequantGemm(gpu.gridScheduler, 256, gpu.dim, 6144);
        PlanDispatchEvidence.assertQwen35BatchDeltaRuleWarp(gpu.gridScheduler, 48, 128);
        PlanDispatchEvidence.assertQwen35FfnDownOnDequantGemm(
                gpu.gridScheduler, 256, gpu.dim, 17408);
    }
}
