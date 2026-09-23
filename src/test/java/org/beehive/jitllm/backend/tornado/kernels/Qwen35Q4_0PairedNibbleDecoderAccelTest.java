package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * Experiment: {@code dequantizeQ4_0ToFP16TiledPairs} — both nibbles of a packed byte decoded by one
 * lane into the tiled layout — against the retained one-element-per-lane decoder.
 *
 * <p>The address mapping is proved on the host first: over every lane of several shapes, the low
 * and high half positions the candidate writes are a bijection onto the matrix's half positions,
 * the retained layout's inverse returns them to the same row at elements {@code 32b + t} and {@code
 * 32b + t + 16}, and the largest address is {@code n * k - 1}. Then on the device: every decoded
 * half bit-equal to the retained decoder's over every nibble, both scale signs, scales across the
 * finite range, block/row/tile boundaries and the production shapes, into a NaN-poisoned
 * destination; a swapped-nibble candidate and a misplaced-high-half candidate both differ. An
 * opt-in screen times decoder alone and decoder + unchanged {@code gemmMMATiledB}.
 */
// @formatter:on
public class Qwen35Q4_0PairedNibbleDecoderAccelTest {

    private static final int BLOCK_BYTES = 18;
    private static final int SCRATCH_ELEMENTS = 17408 * 5120;

    /** The candidate's own lane-to-address mapping, restated on the host. */
    static int[] candidateHalves(int k, int lane) {
        int kSteps = k / 16;
        int parity = lane & 1;
        int pairInSub = ((lane >>> 1) & 1) | (((lane >>> 5) & 1) << 1);
        int kk = ((lane >>> 2) & 7) | (((lane >>> 6) & 1) << 3);
        int sub = (lane >>> 7) & 15;
        int tile = (lane >>> 11) << 1;
        int idx = (sub << 6) + (kk << 2) + pairInSub;
        int lowHalf = (tile << 11) + (idx << 1) + parity;
        return new int[] {
            lowHalf, lowHalf + 2048, tile / kSteps, tile % kSteps, kk, sub, pairInSub, parity
        };
    }

    @Test
    public void theMappingCoversEveryHalfOnceAndPairsTheRightElements() {
        for (int[] shape : new int[][] {{128, 32}, {256, 64}, {384, 1024}, {128, 5120}}) {
            int n = shape[0];
            int k = shape[1];
            boolean[] seen = new boolean[n * k];
            for (int lane = 0; lane < n * k / 2; lane++) {
                int[] c = candidateHalves(k, lane);
                for (int which = 0; which < 2; which++) {
                    int h = c[which];
                    assertTrue("address in range: " + h, h >= 0 && h < n * k);
                    assertTrue("address written once: " + h, !seen[h]);
                    seen[h] = true;
                }
                int[] lo = Qwen35Q4_0TiledDequantGemmAccelTest.tiledElement(k, c[0]);
                int[] hi = Qwen35Q4_0TiledDequantGemmAccelTest.tiledElement(k, c[1]);
                assertEquals("same row", lo[0], hi[0]);
                assertEquals("elements 16 apart", lo[1] + 16, hi[1]);
                assertEquals("low element in the low half of its block", lo[1] % 32, lo[1] % 16);
                int row = c[2] * 128 + (c[5] << 3) + (c[6] << 1) + c[7];
                int element = c[3] * 16 + c[4];
                assertEquals("kernel row", lo[0], row);
                assertEquals("kernel element", lo[1], element);
            }
            for (int h = 0; h < n * k; h++) {
                assertTrue("half " + h + " never written", seen[h]);
            }
        }
    }

    private static byte[] everyNibbleWeights(int n, int k) {
        int blocksPerRow = k / 32;
        byte[] raw = new byte[n * blocksPerRow * BLOCK_BYTES];
        for (int c = 0; c < n; c++) {
            for (int blk = 0; blk < blocksPerRow; blk++) {
                int base = (c * blocksPerRow + blk) * BLOCK_BYTES;
                int bits = 0x2C00 | ((c + blk) & 0xFF);
                if (((c + blk) & 1) == 1) {
                    bits |= 0x8000;
                }
                raw[base] = (byte) (bits & 0xFF);
                raw[base + 1] = (byte) (bits >> 8);
                for (int b = 0; b < 16; b++) {
                    int lo = (b + c + blk) & 0xF;
                    int hi = (15 - b + c) & 0xF;
                    raw[base + 2 + b] = (byte) ((hi << 4) | lo);
                }
            }
        }
        return raw;
    }

    /** Random nibbles; scales spanning the finite half range, both signs, tiny and large. */
    private static byte[] variedScaleWeights(int n, int k, long seed) {
        int blocksPerRow = k / 32;
        byte[] raw = new byte[n * blocksPerRow * BLOCK_BYTES];
        Random rng = new Random(seed);
        rng.nextBytes(raw);
        for (int b = 0; b < n * blocksPerRow; b++) {
            int base = b * BLOCK_BYTES;
            // Exponent 1..26: no subnormals, and scale * 8 stays below the half maximum.
            int exponent = 1 + rng.nextInt(26);
            int bits = (exponent << 10) | rng.nextInt(1024) | (rng.nextBoolean() ? 0x8000 : 0);
            raw[base] = (byte) (bits & 0xFF);
            raw[base + 1] = (byte) (bits >> 8);
        }
        return raw;
    }

    private static byte[] randomWeights(int n, int k, long seed) {
        int blocksPerRow = k / 32;
        byte[] raw = new byte[n * blocksPerRow * BLOCK_BYTES];
        new Random(seed).nextBytes(raw);
        for (int b = 0; b < n * blocksPerRow; b++) {
            int base = b * BLOCK_BYTES;
            int bits = 0x2C00 | (b & 0xFF);
            if ((b & 1) == 1) {
                bits |= 0x8000;
            }
            raw[base] = (byte) (bits & 0xFF);
            raw[base + 1] = (byte) (bits >> 8);
        }
        return raw;
    }

    private static ByteArray toDevice(byte[] raw) {
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }
        return w;
    }

    private static WorkerGrid lanes(long count) {
        WorkerGrid g = new WorkerGrid1D((int) count);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    private static WorkerGrid gemmGrid(int m, int n) {
        WorkerGrid g = new WorkerGrid2D((m / 128) * 256, n / 128);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    enum Candidate {
        PAIRS,
        SWAPPED,
        MISPLACED
    }

    /** Both decoders over the same weights; returns the number of halves that differ. */
    private static int decodeMismatches(String what, int n, int k, byte[] raw, Candidate which)
            throws Exception {
        ByteArray w = toDevice(raw);
        HalfFloatArray control = new HalfFloatArray(n * k);
        HalfFloatArray candidate = new HalfFloatArray(n * k);
        control.init(new HalfFloat(Float.NaN));
        candidate.init(new HalfFloat(Float.NaN));
        TaskGraph graph =
                new TaskGraph("dq")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, w, control, candidate)
                        .task(
                                "control",
                                Qwen35ReferenceKernels::dequantizeQ4_0ToFP16Tiled,
                                new KernelContext(),
                                w,
                                control,
                                n,
                                k)
                        .task(
                                "candidate",
                                which == Candidate.PAIRS
                                        ? Qwen35MMAKernels::dequantizeQ4_0ToFP16TiledPairs
                                        : which == Candidate.SWAPPED
                                                ? Qwen35Q4_0PairedNibbleDecoderAccelTest
                                                        ::pairsWithTheNibblesSwapped
                                                : Qwen35Q4_0PairedNibbleDecoderAccelTest
                                                        ::pairsWithTheHighHalfMisplaced,
                                new KernelContext(),
                                w,
                                candidate,
                                n,
                                k)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, control, candidate);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("dq.control", lanes((long) n * k));
        s.addWorkerGrid("dq.candidate", lanes((long) n * k / 2));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        int mismatches = 0;
        String first = null;
        for (int h = 0; h < n * k; h++) {
            short c = control.get(h).getHalfFloatValue();
            short d = candidate.get(h).getHalfFloatValue();
            assertTrue(what + ": control not finite at " + h, (c & 0x7C00) != 0x7C00);
            if (which == Candidate.PAIRS) {
                assertTrue(
                        what
                                + ": candidate not finite at "
                                + h
                                + ": 0x"
                                + Integer.toHexString(d & 0xFFFF),
                        (d & 0x7C00) != 0x7C00);
            }
            if (c != d) {
                if (first == null) {
                    int[] e = Qwen35Q4_0TiledDequantGemmAccelTest.tiledElement(k, h);
                    first =
                            "half "
                                    + h
                                    + " (row "
                                    + e[0]
                                    + " element "
                                    + e[1]
                                    + "): control 0x"
                                    + Integer.toHexString(c & 0xFFFF)
                                    + " candidate 0x"
                                    + Integer.toHexString(d & 0xFFFF);
                }
                mismatches++;
            }
        }
        if (which == Candidate.PAIRS) {
            assertEquals(
                    what + ": " + mismatches + " halves differ, first at " + first, 0, mismatches);
        }
        return mismatches;
    }

    @Test
    public void everyNibbleAndBothScaleSignsDecodeToTheRetainedBits() throws Exception {
        decodeMismatches(
                "nibbles 256x512", 256, 512, everyNibbleWeights(256, 512), Candidate.PAIRS);
        decodeMismatches("nibbles 128x32", 128, 32, everyNibbleWeights(128, 32), Candidate.PAIRS);
    }

    @Test
    public void scalesAcrossTheFiniteRangeDecodeToTheRetainedBits() throws Exception {
        decodeMismatches(
                "varied 384x1024", 384, 1024, variedScaleWeights(384, 1024, 3L), Candidate.PAIRS);
        decodeMismatches(
                "varied 256x64", 256, 64, variedScaleWeights(256, 64, 4L), Candidate.PAIRS);
    }

    @Test
    public void theProductionShapesDecodeToTheRetainedBits() throws Exception {
        decodeMismatches("gate/up", 17408, 5120, randomWeights(17408, 5120, 11L), Candidate.PAIRS);
        decodeMismatches("ffn_down", 5120, 17408, randomWeights(5120, 17408, 12L), Candidate.PAIRS);
        decodeMismatches("ssm_qkv", 10240, 5120, randomWeights(10240, 5120, 13L), Candidate.PAIRS);
    }

    @Test
    public void theBrokenCandidatesDiffer() throws Exception {
        byte[] raw = everyNibbleWeights(256, 512);
        assertTrue(
                "swapped nibbles agreed",
                decodeMismatches("swapped", 256, 512, raw, Candidate.SWAPPED) > 0);
        assertTrue(
                "misplaced high half agreed",
                decodeMismatches("misplaced", 256, 512, raw, Candidate.MISPLACED) > 0);
    }

    /**
     * Decoder alone and decoder + unchanged gemmMMATiledB, control versus candidate, alternating.
     */
    @Test
    public void screen() throws Exception {
        assumeTrue(
                "opt in with JITLLM_KERNEL_SCREEN=true",
                Boolean.getBoolean("jitllm.kernelScreen")
                        || "true".equals(System.getenv("JITLLM_KERNEL_SCREEN")));
        HalfFloatArray scratch = new HalfFloatArray(SCRATCH_ELEMENTS);
        int[][] shapes = {{17408, 5120}, {5120, 17408}, {10240, 5120}};
        String[] names = {"gate/up", "ffn_down", "ssm_qkv"};
        for (int m : new int[] {128, 256, 512, 1024}) {
            for (int si = 0; si < shapes.length; si++) {
                int n = shapes[si][0];
                int k = shapes[si][1];
                ByteArray w = toDevice(randomWeights(n, k, 7L + n));
                HalfFloatArray a = new HalfFloatArray(m * k);
                Random rng = new Random(9L + m);
                for (int i = 0; i < m * k; i++) {
                    a.set(i, new HalfFloat(rng.nextFloat() * 2.0f - 1.0f));
                }
                FloatArray o1 = new FloatArray(m * n);
                FloatArray o2 = new FloatArray(m * n);
                TaskGraph dc =
                        new TaskGraph("dc")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, w, scratch)
                                .task(
                                        "d",
                                        Qwen35ReferenceKernels::dequantizeQ4_0ToFP16Tiled,
                                        new KernelContext(),
                                        w,
                                        scratch,
                                        n,
                                        k)
                                .transferToHost(DataTransferMode.UNDER_DEMAND, scratch);
                TaskGraph dk =
                        new TaskGraph("dk")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, w, scratch)
                                .task(
                                        "d",
                                        Qwen35MMAKernels::dequantizeQ4_0ToFP16TiledPairs,
                                        new KernelContext(),
                                        w,
                                        scratch,
                                        n,
                                        k)
                                .transferToHost(DataTransferMode.UNDER_DEMAND, scratch);
                TaskGraph pc =
                        new TaskGraph("pc")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, w, scratch)
                                .task(
                                        "d",
                                        Qwen35ReferenceKernels::dequantizeQ4_0ToFP16Tiled,
                                        new KernelContext(),
                                        w,
                                        scratch,
                                        n,
                                        k)
                                .task(
                                        "g",
                                        Qwen35MMAKernels::gemmMMATiledB,
                                        new KernelContext(),
                                        a,
                                        scratch,
                                        o1,
                                        m,
                                        n,
                                        k)
                                .transferToHost(DataTransferMode.UNDER_DEMAND, o1);
                TaskGraph pk =
                        new TaskGraph("pk")
                                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, w, scratch)
                                .task(
                                        "d",
                                        Qwen35MMAKernels::dequantizeQ4_0ToFP16TiledPairs,
                                        new KernelContext(),
                                        w,
                                        scratch,
                                        n,
                                        k)
                                .task(
                                        "g",
                                        Qwen35MMAKernels::gemmMMATiledB,
                                        new KernelContext(),
                                        a,
                                        scratch,
                                        o2,
                                        m,
                                        n,
                                        k)
                                .transferToHost(DataTransferMode.UNDER_DEMAND, o2);
                GridScheduler sdc = new GridScheduler();
                sdc.addWorkerGrid("dc.d", lanes((long) n * k));
                GridScheduler sdk = new GridScheduler();
                sdk.addWorkerGrid("dk.d", lanes((long) n * k / 2));
                GridScheduler spc = new GridScheduler();
                spc.addWorkerGrid("pc.d", lanes((long) n * k));
                spc.addWorkerGrid("pc.g", gemmGrid(m, n));
                GridScheduler spk = new GridScheduler();
                spk.addWorkerGrid("pk.d", lanes((long) n * k / 2));
                spk.addWorkerGrid("pk.g", gemmGrid(m, n));
                try (TornadoExecutionPlan p1 = new TornadoExecutionPlan(dc.snapshot());
                        TornadoExecutionPlan p2 = new TornadoExecutionPlan(dk.snapshot());
                        TornadoExecutionPlan p3 = new TornadoExecutionPlan(pc.snapshot());
                        TornadoExecutionPlan p4 = new TornadoExecutionPlan(pk.snapshot())) {
                    p1.withGridScheduler(sdc).withProfiler(ProfilerMode.SILENT);
                    p2.withGridScheduler(sdk).withProfiler(ProfilerMode.SILENT);
                    p3.withGridScheduler(spc).withProfiler(ProfilerMode.SILENT);
                    p4.withGridScheduler(spk).withProfiler(ProfilerMode.SILENT);
                    for (int i = 0; i < 5; i++) {
                        p1.execute();
                        p2.execute();
                        p3.execute();
                        p4.execute();
                    }
                    int samples = 15;
                    long[] t1 = new long[samples];
                    long[] t2 = new long[samples];
                    long[] t3 = new long[samples];
                    long[] t4 = new long[samples];
                    for (int i = 0; i < samples; i++) {
                        if ((i & 1) == 0) {
                            t1[i] = kernelNs(p1.execute());
                            t2[i] = kernelNs(p2.execute());
                            t3[i] = kernelNs(p3.execute());
                            t4[i] = kernelNs(p4.execute());
                        } else {
                            t4[i] = kernelNs(p4.execute());
                            t3[i] = kernelNs(p3.execute());
                            t2[i] = kernelNs(p2.execute());
                            t1[i] = kernelNs(p1.execute());
                        }
                    }
                    String tag = " " + names[si] + " m=" + m;
                    report("decoder control " + tag, t1);
                    report("decoder pairs   " + tag, t2);
                    report("pair control    " + tag, t3);
                    report("pair pairs      " + tag, t4);
                }
            }
        }
    }

    private static long kernelNs(TornadoExecutionResult result) {
        return result.getProfilerResult().getDeviceKernelTime();
    }

    private static void report(String label, long[] ns) {
        long[] sorted = ns.clone();
        Arrays.sort(sorted);
        StringBuilder samples = new StringBuilder();
        for (long v : ns) {
            samples.append(String.format(Locale.ROOT, "%.1f;", v / 1e3));
        }
        System.out.printf(
                Locale.ROOT,
                "[screen] %-36s min %.1f  median %.1f  max %.1f us  samples(us) %s%n",
                label,
                sorted[0] / 1e3,
                sorted[sorted.length / 2] / 1e3,
                sorted[sorted.length - 1] / 1e3,
                samples);
    }

    // ---- negative controls ---------------------------------------------------------------------

    /** The candidate with the nibbles swapped. */
    public static void pairsWithTheNibblesSwapped(
            KernelContext ctx, ByteArray w, HalfFloatArray out, int n, int k) {
        int lane = ctx.globalIdx;
        int kSteps = k / 16;
        int parity = lane & 1;
        int pairInSub = ((lane >>> 1) & 1) | (((lane >>> 5) & 1) << 1);
        int kk = ((lane >>> 2) & 7) | (((lane >>> 6) & 1) << 3);
        int sub = (lane >>> 7) & 15;
        int tilePair = lane >>> 11;
        int tile = tilePair << 1;
        int idx = (sub << 6) + (kk << 2) + pairInSub;
        int rowBlock = tile / kSteps;
        int kStep = tile - rowBlock * kSteps;
        int row = rowBlock * 128 + (sub << 3) + (pairInSub << 1) + parity;
        int element = kStep * 16 + kk;
        int blocksPerRow = k / 32;
        int block = element >> 5;
        int within = element & 15;
        int base = (row * blocksPerRow + block) * 18;
        float scale = w.getHalfFloat(base).getFloat32();
        int packed = w.get(base + 2 + within) & 0xFF;
        int high = packed & 0xF; // NEGATIVE CONTROL: swapped
        int low = (packed & 0xF0) >>> 4;
        int lowHalf = (tile << 11) + (idx << 1) + parity;
        out.set(lowHalf, new HalfFloat(scale * (low - 8)));
        out.set(lowHalf + 2 * 1024, new HalfFloat(scale * (high - 8)));
    }

    /** The candidate writing the high half one tile too near. */
    public static void pairsWithTheHighHalfMisplaced(
            KernelContext ctx, ByteArray w, HalfFloatArray out, int n, int k) {
        int lane = ctx.globalIdx;
        int kSteps = k / 16;
        int parity = lane & 1;
        int pairInSub = ((lane >>> 1) & 1) | (((lane >>> 5) & 1) << 1);
        int kk = ((lane >>> 2) & 7) | (((lane >>> 6) & 1) << 3);
        int sub = (lane >>> 7) & 15;
        int tilePair = lane >>> 11;
        int tile = tilePair << 1;
        int idx = (sub << 6) + (kk << 2) + pairInSub;
        int rowBlock = tile / kSteps;
        int kStep = tile - rowBlock * kSteps;
        int row = rowBlock * 128 + (sub << 3) + (pairInSub << 1) + parity;
        int element = kStep * 16 + kk;
        int blocksPerRow = k / 32;
        int block = element >> 5;
        int within = element & 15;
        int base = (row * blocksPerRow + block) * 18;
        float scale = w.getHalfFloat(base).getFloat32();
        int packed = w.get(base + 2 + within) & 0xFF;
        int low = packed & 0xF;
        int high = (packed & 0xF0) >>> 4;
        int lowHalf = (tile << 11) + (idx << 1) + parity;
        out.set(lowHalf, new HalfFloat(scale * (low - 8)));
        out.set(
                lowHalf + 1024, // NEGATIVE CONTROL: one tile too near
                new HalfFloat(scale * (high - 8)));
    }
}
