package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.util.Random;
import org.beehive.jitllm.backend.tornado.MlxPrefill;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.mlx.Mlx;

/**
 * The MLX projections of a batched prefill whose FP16 input or product passes FP16's 65504 on one
 * row, as Granite 4.0's first token does in the down projection (134340 into the GEMM, 102097 out
 * of it). With the row scaled, the residual matches an FP32 reference and every row stays finite;
 * the unscaled kernels turn the same row non-finite.
 */
public class MlxPrefillProjectionOverflowAccelTest {

    private static final int ROWS = 4;
    private static final int DIM = 256;
    private static final float RESIDUAL_SCALE = 0.25f;

    /** {@code swigluRowScale}, {@code swigluScaled}, down GEMM, {@code addRowScaledResidual}. */
    @Test
    public void anOverflowingDownProjectionRowIsScaledIntoHalfRangeAndBack() throws Exception {
        assumeTrue("environment absent: MLX on Metal", MlxPrefill.available());
        int hidden = 4096;
        Random rng = new Random(7L);
        HalfFloatArray gateUp = new HalfFloatArray(ROWS * 2 * hidden);
        float[] h = new float[ROWS * hidden];
        for (int b = 0; b < ROWS; b++) {
            for (int i = 0; i < hidden; i++) {
                float gate = (rng.nextFloat() * 2.0f - 1.0f) * 4.0f;
                float up = (rng.nextFloat() * 2.0f - 1.0f) * 4.0f;
                if (b == 0 && i % 512 == 0) {
                    gate = 300.0f;
                    up = 450.0f;
                }
                gateUp.set(b * 2 * hidden + i, new HalfFloat(gate));
                gateUp.set(b * 2 * hidden + hidden + i, new HalfFloat(up));
                gate = gateUp.get(b * 2 * hidden + i).getFloat32();
                up = gateUp.get(b * 2 * hidden + hidden + i).getFloat32();
                h[b * hidden + i] = gate / (1.0f + (float) Math.exp(-gate)) * up;
            }
        }
        HalfFloatArray w = weights(rng, hidden);
        FloatArray x = residual(rng);
        FloatArray xControl = copy(x);
        float[] expected = reference(h, w, x, hidden);

        FloatArray rowScale = new FloatArray(ROWS);
        rowScale.init(Float.NaN);
        HalfFloatArray hb = new HalfFloatArray(ROWS * hidden);
        HalfFloatArray out = new HalfFloatArray(ROWS * DIM);
        HalfFloatArray hbControl = new HalfFloatArray(ROWS * hidden);
        HalfFloatArray outControl = new HalfFloatArray(ROWS * DIM);
        KernelContext context = new KernelContext();
        TaskGraph g =
                new TaskGraph("proj")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                context,
                                gateUp,
                                w,
                                x,
                                xControl,
                                rowScale,
                                hb,
                                out,
                                hbControl,
                                outControl)
                        .task(
                                "row_scale",
                                MlxPrefillKernels::swigluRowScale,
                                context,
                                gateUp,
                                rowScale,
                                hidden,
                                ROWS,
                                256)
                        .task(
                                "half",
                                MlxPrefillKernels::swigluScaled,
                                context,
                                gateUp,
                                hb,
                                rowScale,
                                hidden,
                                ROWS * hidden)
                        .libraryTask("gemm", Mlx::matmulTransposed, hb, w, out, ROWS, hidden, DIM)
                        .task(
                                "residual",
                                MlxPrefillKernels::addRowScaledResidual,
                                context,
                                x,
                                out,
                                rowScale,
                                RESIDUAL_SCALE,
                                DIM,
                                ROWS * DIM)
                        .task(
                                "half_control",
                                MlxPrefillKernels::swiglu,
                                context,
                                gateUp,
                                hbControl,
                                hidden,
                                ROWS * hidden)
                        .libraryTask(
                                "gemm_control",
                                Mlx::matmulTransposed,
                                hbControl,
                                w,
                                outControl,
                                ROWS,
                                hidden,
                                DIM)
                        .task(
                                "residual_control",
                                MlxPrefillKernels::addResidual,
                                context,
                                xControl,
                                outControl,
                                RESIDUAL_SCALE,
                                ROWS * DIM)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, x, xControl, rowScale);
        execute(g, hidden);
        assertScaledRowRecovered(rowScale, xControl, expected, x);
    }

    /** {@code rowScale}, {@code toHalfScaled}, output GEMM, {@code addRowScaledResidual}. */
    @Test
    public void anOverflowingAttentionOutputRowIsScaledIntoHalfRangeAndBack() throws Exception {
        assumeTrue("environment absent: MLX on Metal", MlxPrefill.available());
        int qDim = 2048;
        Random rng = new Random(11L);
        FloatArray xb = new FloatArray(ROWS * qDim);
        float[] in = new float[ROWS * qDim];
        for (int b = 0; b < ROWS; b++) {
            for (int i = 0; i < qDim; i++) {
                float v = (rng.nextFloat() * 2.0f - 1.0f) * 4.0f;
                if (b == 0 && i % 256 == 0) {
                    v = i % 512 == 0 ? 120000.0f : -90000.0f;
                }
                xb.set(b * qDim + i, v);
                in[b * qDim + i] = v;
            }
        }
        HalfFloatArray w = weights(rng, qDim);
        FloatArray x = residual(rng);
        FloatArray xControl = copy(x);
        float[] expected = reference(in, w, x, qDim);

        FloatArray rowScale = new FloatArray(ROWS);
        rowScale.init(Float.NaN);
        HalfFloatArray half = new HalfFloatArray(ROWS * qDim);
        HalfFloatArray out = new HalfFloatArray(ROWS * DIM);
        HalfFloatArray halfControl = new HalfFloatArray(ROWS * qDim);
        HalfFloatArray outControl = new HalfFloatArray(ROWS * DIM);
        KernelContext context = new KernelContext();
        TaskGraph g =
                new TaskGraph("proj")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                context,
                                xb,
                                w,
                                x,
                                xControl,
                                rowScale,
                                half,
                                out,
                                halfControl,
                                outControl)
                        .task(
                                "row_scale",
                                MlxPrefillKernels::rowScale,
                                context,
                                xb,
                                rowScale,
                                qDim,
                                ROWS,
                                256)
                        .task(
                                "half",
                                MlxPrefillKernels::toHalfScaled,
                                context,
                                xb,
                                half,
                                rowScale,
                                qDim,
                                ROWS * qDim)
                        .libraryTask("gemm", Mlx::matmulTransposed, half, w, out, ROWS, qDim, DIM)
                        .task(
                                "residual",
                                MlxPrefillKernels::addRowScaledResidual,
                                context,
                                x,
                                out,
                                rowScale,
                                RESIDUAL_SCALE,
                                DIM,
                                ROWS * DIM)
                        .task(
                                "half_control",
                                MlxPrefillKernels::toHalf,
                                context,
                                xb,
                                halfControl,
                                ROWS * qDim)
                        .libraryTask(
                                "gemm_control",
                                Mlx::matmulTransposed,
                                halfControl,
                                w,
                                outControl,
                                ROWS,
                                qDim,
                                DIM)
                        .task(
                                "residual_control",
                                MlxPrefillKernels::addResidual,
                                context,
                                xControl,
                                outControl,
                                RESIDUAL_SCALE,
                                ROWS * DIM)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, x, xControl, rowScale);
        execute(g, qDim);
        assertScaledRowRecovered(rowScale, xControl, expected, x);
    }

    /** {@code DIM × k} FP16 weights, with the columns of the row-0 outliers at 0.25. */
    private static HalfFloatArray weights(Random rng, int k) {
        HalfFloatArray w = new HalfFloatArray(DIM * k);
        for (int j = 0; j < DIM; j++) {
            for (int i = 0; i < k; i++) {
                float v = (rng.nextFloat() * 2.0f - 1.0f) * 0.02f;
                w.set(j * k + i, new HalfFloat(i % 512 == 0 ? 0.25f : v));
            }
        }
        return w;
    }

    private static FloatArray residual(Random rng) {
        FloatArray x = new FloatArray(ROWS * DIM);
        for (int i = 0; i < ROWS * DIM; i++) {
            x.set(i, (rng.nextFloat() * 2.0f - 1.0f) * 10.0f);
        }
        return x;
    }

    private static FloatArray copy(FloatArray x) {
        FloatArray c = new FloatArray(x.getSize());
        for (int i = 0; i < x.getSize(); i++) {
            c.set(i, x.get(i));
        }
        return c;
    }

    /** {@code x + RESIDUAL_SCALE * W · in} in FP32, over {@code k} input columns. */
    private static float[] reference(float[] in, HalfFloatArray w, FloatArray x, int k) {
        float[] y = new float[ROWS * DIM];
        for (int b = 0; b < ROWS; b++) {
            for (int j = 0; j < DIM; j++) {
                double sum = 0.0;
                for (int i = 0; i < k; i++) {
                    sum += (double) w.get(j * k + i).getFloat32() * in[b * k + i];
                }
                y[b * DIM + j] = x.get(b * DIM + j) + RESIDUAL_SCALE * (float) sum;
            }
        }
        return y;
    }

    private static void execute(TaskGraph g, int k) {
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("proj.row_scale", grid(ROWS * 256, 256));
        s.addWorkerGrid("proj.half", grid(ROWS * k, 256));
        s.addWorkerGrid("proj.residual", grid(ROWS * DIM, 256));
        s.addWorkerGrid("proj.half_control", grid(ROWS * k, 256));
        s.addWorkerGrid("proj.residual_control", grid(ROWS * DIM, 256));
        try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
            p.withGridScheduler(s).execute();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * Row 0 alone is scaled, the unscaled control overflows on it, and every row of {@code x}
     * matches {@code expected} to FP16 accuracy.
     */
    private static void assertScaledRowRecovered(
            FloatArray rowScale, FloatArray xControl, float[] expected, FloatArray x) {
        assertTrue("row 0 scaled: " + rowScale.get(0), rowScale.get(0) < 1.0f);
        for (int b = 1; b < ROWS; b++) {
            assertEquals("row " + b + " left alone", 1.0f, rowScale.get(b), 0.0f);
        }
        boolean controlOverflowed = false;
        for (int j = 0; j < DIM; j++) {
            controlOverflowed |= !Float.isFinite(xControl.get(j));
        }
        assertTrue(
                "the unscaled control stayed finite: the case no longer overflows",
                controlOverflowed);
        for (int b = 0; b < ROWS; b++) {
            float peak = 0.0f;
            for (int j = 0; j < DIM; j++) {
                peak = Math.max(peak, Math.abs(expected[b * DIM + j]));
            }
            for (int j = 0; j < DIM; j++) {
                int id = b * DIM + j;
                assertTrue("row " + b + " col " + j + " non-finite", Float.isFinite(x.get(id)));
                assertEquals(
                        "row " + b + " col " + j, expected[id], x.get(id), 4e-3f * peak + 1e-2f);
            }
        }
    }

    private static WorkerGrid grid(int global, int local) {
        WorkerGrid w = new WorkerGrid1D(global);
        w.setLocalWork(local, 1, 1);
        return w;
    }
}
