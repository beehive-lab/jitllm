package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;

import java.util.Random;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The Q4_K, Q6_K and packed Q4_0 to FP16 decodes on the device against the host decode, element by
 * element.
 */
public class KQuantDequantFP16AccelTest {

    private static final int BLOCKS = 8;

    private static final int N = BLOCKS * 256;

    private static ByteArray randomBlocks(int blockBytes, int scaleOffset, long seed) {
        return randomBlocks(BLOCKS, blockBytes, scaleOffset, seed);
    }

    private static ByteArray randomBlocks(int blocks, int blockBytes, int scaleOffset, long seed) {
        byte[] raw = new byte[blocks * blockBytes];
        new Random(seed).nextBytes(raw);
        for (int b = 0; b < blocks; b++) {
            int base = b * blockBytes + scaleOffset;
            raw[base] = (byte) (b * 13);
            raw[base + 1] = (byte) (0x24 | (b & 1) << 7);
            if (scaleOffset == 0) {
                raw[base + 2] = (byte) (b * 7);
                raw[base + 3] = (byte) 0x20;
            }
        }
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }
        return w;
    }

    private static HalfFloatArray run(String name, ByteArray w, boolean q4k) {
        HalfFloatArray out = new HalfFloatArray(N);
        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph(name)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, w)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, out);
        if (q4k) {
            graph.task(
                    "decode",
                    TransformerComputeKernelsQ4_K::dequantizeQ4_KToFP16,
                    context,
                    w,
                    out,
                    0,
                    N);
        } else {
            graph.task(
                    "decode",
                    TransformerComputeKernelsQ6_K::dequantizeQ6_KToFP16,
                    context,
                    w,
                    out,
                    0,
                    N);
        }
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        WorkerGrid1D grid = new WorkerGrid1D(N);
        grid.setLocalWork(256, 1, 1);
        GridScheduler scheduler = new GridScheduler(name + ".decode", grid);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return out;
    }

    @Test
    public void q4kMatchesHost() {
        ByteArray w = randomBlocks(144, 0, 1);
        HalfFloatArray out = run("q4k", w, true);
        for (int i = 0; i < N; i++) {
            float expected = TransformerComputeKernelsQ4_K.decode(w, (i / 256) * 144, i % 256);
            assertEquals(
                    "element " + i,
                    new HalfFloat(expected).getFloat32(),
                    out.get(i).getFloat32(),
                    0f);
        }
    }

    @Test
    public void q6kMatchesHost() {
        ByteArray w = randomBlocks(210, 208, 2);
        HalfFloatArray out = run("q6k", w, false);
        for (int i = 0; i < N; i++) {
            float expected = TransformerComputeKernelsQ6_K.decode(w, (i / 256) * 210, i % 256);
            assertEquals(
                    "element " + i,
                    new HalfFloat(expected).getFloat32(),
                    out.get(i).getFloat32(),
                    0f);
        }
    }

    @Test
    public void packedQ4_0MatchesHost() {
        int rows = 256;
        int cols = 128;
        byte[] raw = new byte[rows * (cols / 32) * 18];
        new Random(3).nextBytes(raw);
        for (int b = 0; b < rows * cols / 32; b++) {
            raw[b * 18] = (byte) (b * 11);
            raw[b * 18 + 1] = (byte) (0x24 | (b & 1) << 7);
        }
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }
        ByteArray packed = ByteArray.fromArray(PackedTilePacker.packQ4_0(w, rows, cols));
        int n = rows * cols;
        HalfFloatArray out = new HalfFloatArray(n);
        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("q4p")
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, packed, out)
                        .task(
                                "decode",
                                TransformerComputeKernelsQ4_0::dequantizeQ4_0PackedToFP16,
                                context,
                                packed,
                                out,
                                0,
                                n,
                                cols)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        WorkerGrid1D grid = new WorkerGrid1D(n);
        grid.setLocalWork(256, 1, 1);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(new GridScheduler("q4p.decode", grid)).execute();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        for (int i = 0; i < n; i++) {
            float expected = TransformerComputeKernelsQ4_0.decode(w, (i / 32) * 18, i % 32);
            assertEquals(
                    "element " + i,
                    new HalfFloat(expected).getFloat32(),
                    out.get(i).getFloat32(),
                    0f);
        }
    }

    /**
     * The decode-once forms: with the flag clear they write what the per-chunk decode writes; with
     * it set they leave the copy as it was.
     */
    @Test
    public void theDecodeOnceFormsDecodeOnlyUntilTheFlagIsSet() {
        for (boolean q4k : new boolean[] {true, false}) {
            ByteArray w = q4k ? randomBlocks(144, 0, 5) : randomBlocks(210, 208, 6);
            HalfFloatArray expected = run(q4k ? "q4kRef" : "q6kRef", w, q4k);
            for (int flagValue : new int[] {0, 1}) {
                HalfFloatArray out = new HalfFloatArray(N);
                out.init(new HalfFloat(-7.0f));
                IntArray done = new IntArray(1);
                done.set(0, flagValue);
                String name = (q4k ? "q4kOnce" : "q6kOnce") + flagValue;
                TaskGraph graph =
                        new TaskGraph(name)
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, w, out, done);
                if (q4k) {
                    graph.task(
                            "decode",
                            TransformerComputeKernelsQ4_K::dequantizeQ4_KToFP16Once,
                            new KernelContext(),
                            w,
                            out,
                            0,
                            N,
                            done,
                            0);
                } else {
                    graph.task(
                            "decode",
                            TransformerComputeKernelsQ6_K::dequantizeQ6_KToFP16Once,
                            new KernelContext(),
                            w,
                            out,
                            0,
                            N,
                            done,
                            0);
                }
                graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
                WorkerGrid1D grid = new WorkerGrid1D(N);
                grid.setLocalWork(256, 1, 1);
                try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                    plan.withGridScheduler(new GridScheduler(name + ".decode", grid)).execute();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
                for (int i = 0; i < N; i++) {
                    float want = flagValue == 0 ? expected.get(i).getFloat32() : -7.0f;
                    assertEquals(name + " element " + i, want, out.get(i).getFloat32(), 0f);
                }
            }
        }
    }

    /**
     * The tiled forms write, at every half position, the host decode of the element the {@code
     * gemmMMATiledB} tile order puts there, and the positions cover every element once.
     */
    @Test
    public void theTiledFormsWriteTheHostDecodeInTileOrder() {
        int rows = 256;
        int k = 512;
        int n = rows * k;
        int kSteps = k / 16;
        for (boolean q4k : new boolean[] {true, false}) {
            int blockBytes = q4k ? 144 : 210;
            ByteArray w = randomBlocks(n / 256, blockBytes, q4k ? 0 : 208, q4k ? 7 : 8);
            HalfFloatArray out = new HalfFloatArray(n);
            IntArray done = new IntArray(1);
            String name = q4k ? "q4kTiled" : "q6kTiled";
            TaskGraph graph =
                    new TaskGraph(name)
                            .transferToDevice(DataTransferMode.FIRST_EXECUTION, w, out, done);
            if (q4k) {
                graph.task(
                        "decode",
                        Qwen35MMAKernels::dequantizeQ4_KToFP16Tiled,
                        new KernelContext(),
                        w,
                        out,
                        0,
                        n,
                        k,
                        done,
                        0);
            } else {
                graph.task(
                        "decode",
                        Qwen35MMAKernels::dequantizeQ6_KToFP16Tiled,
                        new KernelContext(),
                        w,
                        out,
                        0,
                        n,
                        k,
                        done,
                        0);
            }
            graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
            WorkerGrid1D grid = new WorkerGrid1D(Math.min(n, Qwen35MMAKernels.TILED_DECODE_LANES));
            grid.setLocalWork(256, 1, 1);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.withGridScheduler(new GridScheduler(name + ".decode", grid)).execute();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            boolean[] seen = new boolean[n];
            for (int h = 0; h < n; h++) {
                int tile = h >>> 11;
                int idx = (h >>> 1) & 1023;
                int row = (tile / kSteps) * 128 + ((idx >>> 6) << 3) + ((idx & 3) << 1) + (h & 1);
                int kk = (tile % kSteps) * 16 + ((idx & 63) >>> 2);
                assertEquals(name + " position " + h + " covered twice", false, seen[row * k + kk]);
                seen[row * k + kk] = true;
                int blockByteOffset = (row * (k / 256) + kk / 256) * blockBytes;
                float expected =
                        q4k
                                ? TransformerComputeKernelsQ4_K.decode(w, blockByteOffset, kk % 256)
                                : TransformerComputeKernelsQ6_K.decode(
                                        w, blockByteOffset, kk % 256);
                assertEquals(
                        name + " position " + h,
                        new HalfFloat(expected).getFloat32(),
                        out.get(h).getFloat32(),
                        0f);
            }
        }
    }
}
