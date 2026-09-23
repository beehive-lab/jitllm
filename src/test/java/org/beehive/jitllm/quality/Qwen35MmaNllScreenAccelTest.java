package org.beehive.jitllm.quality;

import static org.junit.Assert.assertEquals;

import org.beehive.jitllm.backend.tornado.PlanDispatchEvidence;
import uk.ac.manchester.tornado.api.GridScheduler;

// @formatter:off
/**
 * {@link Qwen35NllScreenAccelTest} with the tensor-core batched prefill selected.
 *
 * <p>The same passages and the same helper: the unscored prefix is ingested through the batched
 * path and the identical teacher-forced continuation positions are scored. What differs from the
 * parent is only which kernels the prefix went through — `jitllm.qwen35.tensorCores` is set here,
 * before this JVM touches the layer class, so the plan is the MMA one, and {@link #verifyDispatch}
 * checks that against the scheduler of the plan this screen scored with, not against the property.
 *
 * <p>Drive it with {@code -Djitllm.nllScreen.batch=32} for the batched prefix, and with {@code
 * -Djitllm.nllScreen.out=<file>} to write the per-passage report. Without the batch width the plan
 * is the single-token one, which has no batched projection to check, and this class skips rather
 * than scoring a path its name does not describe.
 *
 * <p><b>This is the repository's reused development screen, not independent quality validation.</b>
 * Five passages from this repository's own files, one register, one domain, and it has driven
 * several accept/reject decisions already.
 */
// @formatter:on
public class Qwen35MmaNllScreenAccelTest extends Qwen35NllScreenAccelTest {

    /** Compared against references captured with an FP32 key/value cache. */
    @org.junit.ClassRule
    public static final org.beehive.jitllm.golden.Fp32KeyValueCache FP32_KEY_VALUE_CACHE =
            new org.beehive.jitllm.golden.Fp32KeyValueCache();

    static {
        System.setProperty("jitllm.qwen35.tensorCores", "true");
    }

    @Override
    protected boolean applies(int batch) {
        return batch > 1;
    }

    @Override
    protected void verifyDispatch(GridScheduler grids, int batch, int dim) {
        // At the widths that fill whole GEMM tiles the projection runs as the dequantize-then-GEMM
        // pair; below them, as the direct tensor-core kernel. Either way it is on the tensor cores.
        if (org.beehive.jitllm.model.qwen35.Qwen35Configuration.dequantGemmWidth(batch)) {
            PlanDispatchEvidence.assertQwen35AttentionOutputOnDequantGemm(grids, batch, dim, 6144);
        } else {
            PlanDispatchEvidence.assertQwen35AttentionOutputOnTensorCores(grids, batch, dim);
        }
    }

    @Override
    protected void verifyBatchedScan(String kernel, int stateDim) {
        // The 128-wide state on CUDA takes the warp-per-column scan; the same width elsewhere the
        // shared-state one. The report names the kernel; this pins it to the dispatch rule.
        String expected =
                org.beehive.jitllm.backend.tornado.kernels.Qwen35BatchKernels.deltaWarpEligible(
                                        stateDim)
                                && org.beehive.jitllm.backend.tornado.TensorCoreSupport
                                        .isTensorCoreCapableBackend()
                        ? "deltaRuleScanWarp"
                        : "deltaRuleScanShared";
        assertEquals("batched delta-rule scan this screen scored", expected, kernel);
    }
}
