package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;

import java.util.Set;
import org.beehive.jitllm.backend.tornado.PlanDispatchEvidence;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

// @formatter:off
/**
 * {@link Qwen35MmaBatchedPrefillWidth256ParityAccelTest} with the FP16 key/value cache — the cache
 * the benchmarked plan runs, under which the batched attention is the scored kernel family
 * (currently {@code attentionBatchFP16PagedTensorCore}) rather than the FP32-cache kernel the
 * width-128/256 parity tests exercise. Same fixture, same bounds, same CPU reference; the attention
 * and delta-rule kernels the plan compiled are asserted by name off this run's own plan.
 *
 * <p>The CPU reference computes each attention dot product as an FP32 running sum in dimension
 * order, the order the staged kernels also use; the warp kernel's reassociated sum is at a larger
 * distance from that reference (relL2 0.055 against 0.036 at this width when it was introduced)
 * while being closer to an FP64 reference on the kernel test's cases. That is why this test's
 * metrics sit above the FP32-cache ones; the bounds are the same.
 */
// @formatter:on
public class Qwen35MmaBatchedPrefillWidth256Fp16KvParityAccelTest extends CpuGpuParity {

    static {
        System.setProperty("jitllm.qwen35.tensorCores", "true");
        System.setProperty("jitllm.kvcache.fp16", "true");
    }

    @Test
    public void qwen3_8_27b_q4_0_batchedPrefillParityAt256WithFp16Kv() throws Exception {
        GoldenCapture.Result gpu =
                assertParityBatched(Fixture.QWEN3_8_27B_Q4_0, Q8_0_PACKED_DECODE, 256);
        assertEquals(
                "the batched attention kernel this plan compiled",
                Set.of("attentionBatchFP16PagedTensorCoreT32"),
                gpu.batchedTaskKernels.get("attention"));
        assertEquals(
                "the batched delta-rule scan this plan compiled",
                Set.of("deltaRuleScanWarp"),
                gpu.batchedTaskKernels.get("ssm_delta_rule"));
        PlanDispatchEvidence.assertQwen35AttentionOutputOnDequantGemm(
                gpu.gridScheduler, 256, gpu.dim, 6144);
        PlanDispatchEvidence.assertQwen35SsmOutOnDequantGemm(gpu.gridScheduler, 256, gpu.dim, 6144);
        PlanDispatchEvidence.assertQwen35BatchDeltaRuleWarp(gpu.gridScheduler, 48, 128);
        PlanDispatchEvidence.assertQwen35FfnDownOnDequantGemm(
                gpu.gridScheduler, 256, gpu.dim, 17408);
        assertEquals(
                "the batched alpha/beta projections this plan compiled",
                Set.of("batchedMatVecF32WarpTile"),
                gpu.batchedTaskKernels.get("ssm_alpha_proj"));
    }
}
