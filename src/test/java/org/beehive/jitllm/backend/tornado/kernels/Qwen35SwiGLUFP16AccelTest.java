package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;
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

/**
 * {@code swiGLUBatchFP16} against {@code swiGLUBatch} followed by {@code convertToFP16}: raw-bit
 * equal halves over the production chunk (2048 x 17408) with gates spanning the saturating range of
 * the sigmoid and products around the half format's subnormal and overflow edges, into NaN-poisoned
 * destinations.
 */
public class Qwen35SwiGLUFP16AccelTest {

    @Test
    public void theFusedFormIsRawBitEqualToSwiGLUThenConvert() throws Exception {
        int n = 2048 * 17408;
        FloatArray gate = new FloatArray(n);
        FloatArray up = new FloatArray(n);
        Random rng = new Random(3L);
        for (int i = 0; i < n; i++) {
            int kind = i % 8;
            float g = (rng.nextFloat() * 2.0f - 1.0f) * (kind < 6 ? 8.0f : 120.0f);
            float u =
                    (rng.nextFloat() * 2.0f - 1.0f) * (kind == 7 ? 1e5f : kind == 6 ? 1e-6f : 4.0f);
            gate.set(i, g);
            up.set(i, u);
        }
        FloatArray hb = new FloatArray(n);
        HalfFloatArray control = new HalfFloatArray(n);
        HalfFloatArray candidate = new HalfFloatArray(n);
        hb.init(Float.NaN);
        control.init(new HalfFloat(Float.NaN));
        candidate.init(new HalfFloat(Float.NaN));
        TaskGraph g =
                new TaskGraph("sw")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, gate, up, hb, control, candidate)
                        .task(
                                "swiglu",
                                Qwen35MMAKernels::swiGLUBatch,
                                new KernelContext(),
                                gate,
                                up,
                                hb)
                        .task(
                                "convert",
                                Qwen35MMAKernels::convertToFP16,
                                new KernelContext(),
                                hb,
                                control)
                        .task(
                                "fused",
                                Qwen35MMAKernels::swiGLUBatchFP16,
                                new KernelContext(),
                                gate,
                                up,
                                candidate)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, control, candidate);
        GridScheduler s = new GridScheduler();
        for (String t : new String[] {"sw.swiglu", "sw.convert", "sw.fused"}) {
            WorkerGrid w = new WorkerGrid1D(n);
            w.setLocalWork(128, 1, 1);
            s.addWorkerGrid(t, w);
        }
        try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
            p.withGridScheduler(s).execute();
        }
        int mismatches = 0;
        int nonFinite = 0;
        String first = null;
        for (int i = 0; i < n; i++) {
            short c = control.get(i).getHalfFloatValue();
            short d = candidate.get(i).getHalfFloatValue();
            if ((c & 0x7C00) == 0x7C00) {
                nonFinite++;
            }
            if (c != d) {
                if (first == null) {
                    first =
                            "index "
                                    + i
                                    + ": 0x"
                                    + Integer.toHexString(c & 0xFFFF)
                                    + " vs 0x"
                                    + Integer.toHexString(d & 0xFFFF);
                }
                mismatches++;
            }
        }
        assertTrue("the overflow band produced no non-finite halves", nonFinite > 0);
        assertTrue("every half non-finite", nonFinite < n / 4);
        assertEquals(mismatches + " halves differ, first at " + first, 0, mismatches);
    }
}
