package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.util.Random;
import org.beehive.jitllm.golden.TupleInfo;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * Gemma 4's grouped FP16 decode attention without shuffles ({@link
 * Gemma4AttentionKernels#attentionDecodeGroupFP16Shared}, then {@link
 * Gemma4AttentionKernels#combineDecodeGroup}) against the exact attention computed in FP64 on the
 * host, over the family's flat FP16 cache, its sliding window and both of its head widths.
 *
 * <p>This is the kernel a backend without {@code simdShuffleDown} (OpenCL) runs; it uses nothing
 * but shared memory and barriers, so it runs, and is checked, on every backend. The output is
 * poisoned to NaN first, so an element the kernels do not write fails. The bound, relative L2 at
 * most 1e-5 per case, is FP32 arithmetic over FP16 inputs that the host reads identically; the
 * window one position short is the negative control and must exceed it.
 */
// @formatter:on
public class Gemma4DecodeGroupSharedParityAccelTest {

    private static final int KV_HEADS = 1;
    private static final int HEADS = KV_HEADS * Gemma4AttentionKernels.DECODE_GROUP;
    private static final int CAPACITY = 1100;
    private static final int SLOTS_BEFORE = 2;

    record Case(String name, int headDim, int window, int pos) {}

    private static final Case[] CASES = {
        new Case("sliding, first position", 256, 512, 0),
        new Case("sliding, part of a slice", 256, 512, 20),
        new Case("sliding, past the window", 256, 512, 1000),
        new Case("full, several slices", 512, CAPACITY, 777),
    };

    @Test
    public void matchesTheExactAttention() {
        assumeTrue("no TornadoVM device", TupleInfo.acceleratorPresent());
        for (Case c : CASES) {
            double rel = relativeL2(c, c.window());
            System.out.printf("[gemma4-decode-shared] %s: rel L2 %.3e%n", c.name(), rel);
            assertTrue(c.name() + ": relative L2 " + rel, rel <= 1e-5);
        }
        // Negative control: the device window one position shorter than the reference's.
        Case past = CASES[2];
        double control = relativeL2(past, past.window() - 1);
        System.out.printf("[gemma4-decode-shared] control (window - 1): rel L2 %.3e%n", control);
        assertTrue("the control is not distinguished: " + control, control > 1e-5);
    }

    /** Runs the kernels with {@code deviceWindow} and compares against the exact attention. */
    private static double relativeL2(Case c, int deviceWindow) {
        int headDim = c.headDim();
        int kvDim = KV_HEADS * headDim;
        int base = SLOTS_BEFORE * CAPACITY * kvDim;
        Random random = new Random(0x51ed + headDim + c.pos());

        FloatArray q = new FloatArray(HEADS * headDim);
        for (int i = 0; i < q.getSize(); i++) {
            q.set(i, (float) random.nextGaussian() * 0.25f);
        }
        int cacheSize = base + CAPACITY * kvDim;
        HalfFloatArray keys = new HalfFloatArray(cacheSize);
        HalfFloatArray values = new HalfFloatArray(cacheSize);
        float[] k = new float[cacheSize];
        float[] v = new float[cacheSize];
        for (int i = 0; i < cacheSize; i++) {
            HalfFloat kh = new HalfFloat((float) random.nextGaussian());
            HalfFloat vh = new HalfFloat((float) random.nextGaussian());
            keys.set(i, kh);
            values.set(i, vh);
            k[i] = kh.getFloat32();
            v[i] = vh.getFloat32();
        }
        int maxSlices = Gemma4AttentionKernels.decodeSlices(Math.min(c.window(), CAPACITY));
        FloatArray partial =
                new FloatArray(
                        KV_HEADS * maxSlices * Gemma4AttentionKernels.decodePartialStride(headDim));
        FloatArray out = new FloatArray(HEADS * headDim);
        out.init(Float.NaN);
        IntArray position = new IntArray(1);
        position.set(0, c.pos());

        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("g")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                q,
                                keys,
                                values,
                                partial,
                                out,
                                position)
                        .task(
                                "group",
                                Gemma4AttentionKernels::attentionDecodeGroupFP16Shared,
                                context,
                                q,
                                keys,
                                values,
                                partial,
                                position,
                                headDim,
                                kvDim,
                                base,
                                deviceWindow,
                                maxSlices)
                        .task(
                                "combine",
                                Gemma4AttentionKernels::combineDecodeGroup,
                                context,
                                partial,
                                out,
                                position,
                                HEADS,
                                headDim,
                                deviceWindow,
                                maxSlices)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        WorkerGrid1D group = new WorkerGrid1D(KV_HEADS * maxSlices * 256);
        group.setLocalWork(Gemma4AttentionKernels.DECODE_LANES, 1, 1);
        WorkerGrid1D combine = new WorkerGrid1D(HEADS * headDim);
        combine.setLocalWork(128, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("g.group", group);
        scheduler.addWorkerGrid("g.combine", combine);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        } catch (Exception e) {
            throw new AssertionError(e);
        }

        double num = 0;
        double den = 0;
        int from = Math.max(0, c.pos() - c.window() + 1);
        for (int h = 0; h < HEADS; h++) {
            int kvHead = h / Gemma4AttentionKernels.DECODE_GROUP;
            double[] scores = new double[c.pos() - from + 1];
            double max = Double.NEGATIVE_INFINITY;
            for (int t = from; t <= c.pos(); t++) {
                double s = 0;
                for (int d = 0; d < headDim; d++) {
                    s +=
                            (double) q.get(h * headDim + d)
                                    * k[base + t * kvDim + kvHead * headDim + d];
                }
                scores[t - from] = s;
                max = Math.max(max, s);
            }
            double sum = 0;
            for (int i = 0; i < scores.length; i++) {
                scores[i] = Math.exp(scores[i] - max);
                sum += scores[i];
            }
            for (int d = 0; d < headDim; d++) {
                double exact = 0;
                for (int t = from; t <= c.pos(); t++) {
                    exact += scores[t - from] * v[base + t * kvDim + kvHead * headDim + d];
                }
                exact /= sum;
                double got = out.get(h * headDim + d);
                if (Double.isNaN(got)) {
                    return Double.POSITIVE_INFINITY;
                }
                num += (got - exact) * (got - exact);
                den += exact * exact;
            }
        }
        return Math.sqrt(num / den);
    }
}
