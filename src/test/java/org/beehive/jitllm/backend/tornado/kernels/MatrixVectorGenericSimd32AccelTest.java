package org.beehive.jllm.backend.tornado.kernels;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import org.beehive.jllm.backend.tornado.scheduling.Fp16GemvReductionPolicy;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerDetectionService;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Metal parity, task 6 — isolated correctness for {@link TransformerComputeKernelsLayered
 * #matrixVectorGenericSimd32}, the vocabulary-projection sibling derived from the already-verified
 * {@code matrixVectorGenericWithResidualSimd32} (residual add removed, nothing else changed).
 *
 * <p>Dimensions are representative of the real vocabulary projection's shape — input dimension in
 * Llama-3.2-1B's actual range (2048), output rows an order of magnitude larger (8000, against a
 * real vocabulary of 128256) — deliberately not a square matrix, since a square shape would not
 * catch a row/column transposition the real call site's rectangular shape would.
 */
public class MatrixVectorGenericSimd32AccelTest {

    private static final int N = 2048;
    private static final int D = 8000;
    private static final float POISON = -999999f;

    @Test
    public void matchesTheCpuReferenceWhereThisKernelIsSelectedForVocabularyShapedDimensions() {
        // Wherever the vocabulary projection actually selects this kernel, not just on Metal.
        // It was Metal-only while SUBGROUP_SHUFFLE_32 was the only grant; CUDA now selects it too
        // through Fp16GemvReductionPolicy, and a CPU-reference check that skipped there would
        // leave the shipping path unverified against anything tighter than an end-to-end
        // tolerance. This is the same disjunction LogitsFP16Layer.useSimd32Reduction() evaluates,
        // so the gate follows the selection rather than restating a device name.
        assumeTrue(
                "this device does not select the 32-lane shuffle reduction",
                SchedulerDetectionService.isSubgroupShuffle32Supported()
                        || Fp16GemvReductionPolicy.preferShuffleReduction());

        HalfFloatArray x = new HalfFloatArray(N);
        for (int i = 0; i < N; i++) {
            // Small exact integers: representable in FP16 with no rounding, so the CPU reference
            // has no ambiguity beyond summation order, which stays exact for this value range.
            x.set(i, new HalfFloat((i % 3) - 1));
        }
        HalfFloatArray w = new HalfFloatArray(D * N);
        for (int r = 0; r < D; r++) {
            for (int c = 0; c < N; c++) {
                w.set(r * N + c, new HalfFloat(((r + c) % 4) - 1));
            }
        }
        FloatArray hb = new FloatArray(D);
        for (int i = 0; i < D; i++) {
            hb.set(i, POISON);
        }

        float[] expected = new float[D];
        for (int r = 0; r < D; r++) {
            float sum = 0f;
            for (int c = 0; c < N; c++) {
                sum += w.get(r * N + c).getFloat32() * x.get(c).getFloat32();
            }
            expected[r] = sum;
        }

        KernelContext context = new KernelContext();
        TaskGraph tg =
                new TaskGraph("vocabProjSimd32Probe")
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, x, w, hb)
                        .task(
                                "vocab_proj",
                                TransformerComputeKernelsLayered::matrixVectorGenericSimd32,
                                context,
                                x,
                                hb,
                                w,
                                N,
                                D)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, hb);

        WorkerGrid1D worker = new WorkerGrid1D(D * 32);
        worker.setLocalWork(32, 1, 1);
        GridScheduler gs = new GridScheduler();
        gs.addWorkerGrid("vocabProjSimd32Probe.vocab_proj", worker);

        ImmutableTaskGraph itg = tg.snapshot();
        TornadoExecutionPlan plan = new TornadoExecutionPlan(itg);
        try {
            plan.withGridScheduler(gs);
            plan.execute();
        } finally {
            plan.freeDeviceMemory();
        }

        float maxAbs = 0f;
        float maxRel = 0f;
        boolean poisonRemains = false;
        for (int r = 0; r < D; r++) {
            float actual = hb.get(r);
            if (actual == POISON) {
                poisonRemains = true;
            }
            float diff = Math.abs(actual - expected[r]);
            maxAbs = Math.max(maxAbs, diff);
            maxRel = Math.max(maxRel, diff / (Math.abs(expected[r]) + 1e-6f));
        }
        System.out.println(
                "PROBE matrixVectorGenericSimd32 maxAbsDiff="
                        + maxAbs
                        + " maxRelDiff="
                        + maxRel
                        + " poisonRemains="
                        + poisonRemains);

        assertFalse("the output buffer must not still contain the poison sentinel", poisonRemains);
        // Inputs are small exact integers with no FP16 rounding ambiguity, so exact agreement is
        // the correct bar here, not the project's general atol/rtol numerical-parity bounds
        // (those exist for cross-arithmetic comparisons, not for a kernel matching its own
        // reference on values it represents exactly).
        assertTrue(
                "expected bit-for-bit agreement with the CPU reference for exact-FP16 inputs;"
                        + " maxAbsDiff="
                        + maxAbs
                        + " maxRelDiff="
                        + maxRel,
                maxAbs == 0f);
    }
}
