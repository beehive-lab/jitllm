package org.beehive.jitllm.golden;

import org.beehive.jitllm.backend.tornado.PlanDispatchEvidence;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

// @formatter:off
/**
 * The batched prefill parity of {@link Qwen35BatchedPrefillWidth64ParityAccelTest}, with the
 * **tensor-core path actually selected**, and the selection read off the plan the run built.
 *
 * <p>{@code Qwen35BatchPrefillLayers.TENSOR_CORES} is a static final read from {@code
 * jitllm.qwen35.tensorCores} at class initialization, so the ordinary parity classes — which never
 * set it — build the scalar batched plan and are not coverage of the MMA path. The property here is
 * set in a static initializer, before this JVM touches the layer class, and {@code
 * reuseForks=false} gives the class its own process.
 *
 * <p>That selects the path. The check is {@link
 * PlanDispatchEvidence#assertQwen35AttentionOutputOnTensorCores} against this run's own plan: the
 * conversion, projection and residual tasks of every attention layer, each projection on one warp
 * per {@code BM x BN} output tile.
 *
 * <p>Width 64 is wider than the 63-token fixture: a single chunk, most of it inactive — the active
 * count comes from the chunk, not from the launch width.
 */
// @formatter:on
public class Qwen35MmaBatchedPrefillWidth64ParityAccelTest extends CpuGpuParity {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    static {
        System.setProperty("jitllm.qwen35.tensorCores", "true");
    }

    @Test
    public void qwen3_8_27b_q4_0_batchedPrefillParityAt64OnTensorCores() throws Exception {
        GoldenCapture.Result gpu =
                assertParityBatched(Fixture.QWEN3_8_27B_Q4_0, Q8_0_PACKED_DECODE, 64);
        PlanDispatchEvidence.assertQwen35AttentionOutputOnTensorCores(
                gpu.gridScheduler, 64, gpu.dim);
    }
}
