package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.beehive.jitllm.backend.tornado.TensorCoreSupport;
import org.beehive.jitllm.inference.state.State;
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
 * The batched prefill's tensor-core attention over the paged FP16 cache, as {@code
 * TensorCoreBatchAttention} runs it for a chunk: {@link
 * TensorCoreAttentionKernels#gatherPagedKeysTransposed}, {@link
 * TensorCoreAttentionKernels#queriesToHalf} and the flash kernel for the head width, against a
 * double-precision causal attention over the same FP16 keys, values and queries.
 *
 * <p>Each case is a chunk past position 0 or a first chunk, full or partial, with grouped key/value
 * heads: query row {@code t} sits at position {@code start + t} and must attend to every cached
 * position up to it, the prefix included. The cache holds three sequences in shuffled physical
 * blocks and the chunk reads the third, so the block table and the slot are exercised, and two
 * layers, so the layer offset is.
 *
 * <p>Bound per output: {@code 2^-9 * sum_j p_j |v_j| + 2^-11 * |out|}: the probabilities enter the
 * weighted sum in FP16 and are normalized by a sum kept in FP32 (two half-ulps), with as much again
 * for the scores' FP32 accumulation and the online rescaling, and the output is rounded to FP16.
 * Data checks: references that align the causal mask to the chunk (query {@code t} attends to
 * {@code 0..t}), that read only the chunk's own keys, that map query heads to key/value heads the
 * wrong way, or that read the logical blocks unshuffled must each fall outside it for most outputs.
 * Rows past the chunk's tokens must come out zero.
 */
// @formatter:on
public class TensorCoreBatchAttentionPagedAccelTest {

    /** A chunk: head width, heads, key/value heads, padded rows, start, tokens, context. */
    record Case(
            int headSize, int heads, int kvHeads, int rows, int start, int tokens, int context) {

        int qDim() {
            return heads * headSize;
        }

        int kvDim() {
            return kvHeads * headSize;
        }

        int kvMul() {
            return heads / kvHeads;
        }

        @Override
        public String toString() {
            return "head "
                    + headSize
                    + ", "
                    + heads
                    + "/"
                    + kvHeads
                    + " heads, rows "
                    + rows
                    + ", start "
                    + start
                    + ", tokens "
                    + tokens;
        }
    }

    private static final int LAYERS = 2;
    private static final int LAYER = 1;
    private static final int SLOTS = 3;
    private static final int SLOT = 2;
    private static final int BLOCK = State.KV_BLOCK_SIZE;

    private static final List<Case> CASES =
            List.of(
                    // Qwen3-0.6B's attention shape: a later chunk, full and partial.
                    new Case(128, 16, 8, 128, 300, 128, 640),
                    new Case(128, 16, 8, 128, 517, 37, 640),
                    // A first chunk, partial.
                    new Case(128, 16, 8, 128, 0, 100, 640),
                    // Llama-3.2-1B's (64 wide, four query heads per key/value head).
                    new Case(64, 32, 8, 128, 250, 128, 512),
                    // Phi-3's (96 wide, no grouping), past one tile of prefix.
                    new Case(96, 32, 32, 128, 129, 128, 384),
                    // Two row tiles: a chunk of 129 padded to 256, deep into the cache.
                    new Case(128, 16, 8, 256, 1000, 129, 1280));

    @Test
    public void laterAndPartialChunksAttendToTheirWholePrefix() {
        assumeTrue("tensor-core MMA backend", TensorCoreSupport.isTensorCoreCapableBackend());
        List<String> failures = new ArrayList<>();
        for (Case c : CASES) {
            failures.addAll(run(c));
        }
        assertTrue(String.join("\n", failures), failures.isEmpty());
    }

    private static List<String> run(Case c) {
        Random rng = new Random(0x7016L + 31L * c.start + c.tokens + c.headSize);
        int blocksPerSlot = (c.context + BLOCK - 1) / BLOCK;
        int physical = SLOTS * blocksPerSlot;
        int blockStride = LAYERS * BLOCK * c.kvDim();
        int blockCfg = BLOCK | (blocksPerSlot << 16);
        int paddedPositions = (c.context + 127) / 128 * 128;

        // Every slot's logical blocks spread over the physical ones at random.
        int[] perm = new int[physical];
        for (int i = 0; i < physical; i++) {
            perm[i] = i;
        }
        for (int i = physical - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int t = perm[i];
            perm[i] = perm[j];
            perm[j] = t;
        }
        IntArray table = new IntArray(physical);
        for (int i = 0; i < physical; i++) {
            table.set(i, perm[i]);
        }

        HalfFloatArray keyCache = new HalfFloatArray(physical * blockStride);
        HalfFloatArray valueCache = new HalfFloatArray(physical * blockStride);
        for (int i = 0; i < keyCache.getSize(); i++) {
            keyCache.set(i, new HalfFloat(rng.nextFloat() * 2f - 1f));
            valueCache.set(i, new HalfFloat(rng.nextFloat() * 2f - 1f));
        }
        int qkvStride = c.qDim() + 2 * c.kvDim();
        FloatArray qkv = new FloatArray(c.rows * qkvStride);
        for (int i = 0; i < qkv.getSize(); i++) {
            qkv.set(i, rng.nextFloat() * 4f - 2f);
        }
        IntArray info = new IntArray(3);
        info.set(0, c.start);
        info.set(1, c.tokens);
        info.set(2, SLOT);

        HalfFloatArray queries = new HalfFloatArray(c.rows * c.qDim());
        HalfFloatArray keys = new HalfFloatArray(c.kvDim() * paddedPositions);
        HalfFloatArray valuesT = new HalfFloatArray(c.kvDim() * paddedPositions);
        HalfFloatArray out = new HalfFloatArray(c.rows * c.qDim());
        for (int i = 0; i < out.getSize(); i++) {
            out.set(i, new HalfFloat(Float.NaN));
        }

        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("t16")
                        .transferToDevice(
                                DataTransferMode.FIRST_EXECUTION,
                                info,
                                keyCache,
                                valueCache,
                                table,
                                qkv,
                                queries,
                                keys,
                                valuesT,
                                out)
                        .task(
                                "attn_gather",
                                TensorCoreAttentionKernels::gatherPagedKeysTransposed,
                                context,
                                info,
                                keyCache,
                                valueCache,
                                keys,
                                valuesT,
                                LAYER,
                                c.kvDim(),
                                c.headSize,
                                table,
                                blockCfg,
                                blockStride,
                                paddedPositions)
                        .task(
                                "attn_q16",
                                TensorCoreAttentionKernels::queriesToHalf,
                                context,
                                qkv,
                                queries,
                                c.qDim(),
                                qkvStride,
                                c.rows * c.qDim());
        float scale = (float) (1.0 / Math.sqrt(c.headSize));
        int rowBlocks = c.rows / TensorCoreAttentionKernels.FLASH_ROWS;
        switch (c.headSize) {
            case 64 ->
                    graph.task(
                            "attn_flash",
                            TensorCoreAttentionKernels::flashAttentionHead64,
                            context,
                            info,
                            queries,
                            keys,
                            valuesT,
                            out,
                            c.heads,
                            c.kvMul(),
                            c.qDim(),
                            rowBlocks,
                            paddedPositions,
                            scale);
            case 96 ->
                    graph.task(
                            "attn_flash",
                            TensorCoreAttentionKernels::flashAttentionHead96,
                            context,
                            info,
                            queries,
                            keys,
                            valuesT,
                            out,
                            c.heads,
                            c.kvMul(),
                            c.qDim(),
                            rowBlocks,
                            paddedPositions,
                            scale);
            default ->
                    graph.task(
                            "attn_flash",
                            TensorCoreAttentionKernels::flashAttentionHead128,
                            context,
                            info,
                            queries,
                            keys,
                            valuesT,
                            out,
                            c.heads,
                            c.kvMul(),
                            c.qDim(),
                            rowBlocks,
                            paddedPositions,
                            scale);
        }
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);

        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("t16.attn_gather", elementwise(paddedPositions * c.kvDim()));
        scheduler.addWorkerGrid("t16.attn_q16", elementwise(c.rows * c.qDim()));
        WorkerGrid1D flash =
                new WorkerGrid1D(rowBlocks * c.heads * TensorCoreAttentionKernels.FLASH_THREADS);
        flash.setLocalWork(TensorCoreAttentionKernels.FLASH_THREADS, 1, 1);
        scheduler.addWorkerGrid("t16.attn_flash", flash);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }

        return check(c, keyCache, valueCache, table, blockCfg, blockStride, qkv, out);
    }

    private static WorkerGrid1D elementwise(int n) {
        WorkerGrid1D g = new WorkerGrid1D(n);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    /** Where position {@code pos} of key/value column {@code col} sits, through {@code table}. */
    private static int address(
            IntArray table, int blockCfg, int blockStride, int kvDim, int pos, int col) {
        int blocksPerSlot = blockCfg >>> 16;
        int physical = table.get(SLOT * blocksPerSlot + pos / BLOCK);
        return physical * blockStride + LAYER * BLOCK * kvDim + (pos % BLOCK) * kvDim + col;
    }

    /** The same position read as if the table were the identity. */
    private static int unshuffled(int blockCfg, int blockStride, int kvDim, int pos, int col) {
        int blocksPerSlot = blockCfg >>> 16;
        int physical = SLOT * blocksPerSlot + pos / BLOCK;
        return physical * blockStride + LAYER * BLOCK * kvDim + (pos % BLOCK) * kvDim + col;
    }

    private static List<String> check(
            Case c,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            IntArray table,
            int blockCfg,
            int blockStride,
            FloatArray qkv,
            HalfFloatArray out) {
        int d = c.headSize;
        int kvDim = c.kvDim();
        int qkvStride = c.qDim() + 2 * kvDim;
        int last = c.start + c.tokens;
        // The FP16 keys and values each reference reads: as cached, and unshuffled.
        float[][] k = new float[last][kvDim];
        float[][] v = new float[last][kvDim];
        float[][] kRaw = new float[last][kvDim];
        float[][] vRaw = new float[last][kvDim];
        for (int p = 0; p < last; p++) {
            for (int col = 0; col < kvDim; col++) {
                int a = address(table, blockCfg, blockStride, kvDim, p, col);
                int u = unshuffled(blockCfg, blockStride, kvDim, p, col);
                k[p][col] = keyCache.get(a).getFloat32();
                v[p][col] = valueCache.get(a).getFloat32();
                kRaw[p][col] = keyCache.get(u).getFloat32();
                vRaw[p][col] = valueCache.get(u).getFloat32();
            }
        }
        double scale = 1.0 / Math.sqrt(d);
        int checked = 0;
        int outside = 0;
        double worst = 0;
        int[] wrongOutside = new int[4];
        int wrongChecked = 0;
        String[] wrongNames = {
            "mask aligned to the chunk",
            "the chunk's keys only",
            "heads mapped modulo",
            "unshuffled"
        };
        double[] ref = new double[d];
        double[] mag = new double[d];
        double[][] wrong = new double[4][d];
        for (int t = 0; t < c.tokens; t++) {
            int pos = c.start + t;
            for (int h = 0; h < c.heads; h++) {
                float[] q = new float[d];
                for (int i = 0; i < d; i++) {
                    q[i] = new HalfFloat(qkv.get(t * qkvStride + h * d + i)).getFloat32();
                }
                int g = h / c.kvMul();
                attend(q, k, v, g * d, 0, pos, scale, ref, mag);
                for (int i = 0; i < d; i++) {
                    float got = out.get(t * c.qDim() + h * d + i).getFloat32();
                    double bound = Math.scalb(mag[i], -9) + Math.scalb(Math.abs(ref[i]), -11);
                    double err = Math.abs(got - ref[i]);
                    checked++;
                    if (!Float.isFinite(got) || !(err <= bound)) {
                        outside++;
                    }
                    worst = Math.max(worst, Float.isFinite(got) ? err / bound : Double.MAX_VALUE);
                }
                if (t % 5 == 0 && h % 3 == 0) {
                    attend(q, k, v, g * d, 0, t, scale, wrong[0], new double[d]);
                    attend(q, k, v, g * d, c.start, pos, scale, wrong[1], new double[d]);
                    int gWrong = (h % c.kvHeads) == g ? (g + 1) % c.kvHeads : h % c.kvHeads;
                    attend(q, k, v, gWrong * d, 0, pos, scale, wrong[2], new double[d]);
                    attend(q, kRaw, vRaw, g * d, 0, pos, scale, wrong[3], new double[d]);
                    for (int i = 0; i < d; i++) {
                        double bound = Math.scalb(mag[i], -9) + Math.scalb(Math.abs(ref[i]), -11);
                        wrongChecked++;
                        for (int w = 0; w < 4; w++) {
                            if (!(Math.abs(wrong[w][i] - ref[i]) <= bound)) {
                                wrongOutside[w]++;
                            }
                        }
                    }
                }
            }
        }
        int nonZeroPadding = 0;
        for (int t = c.tokens; t < c.rows; t++) {
            for (int i = 0; i < c.qDim(); i++) {
                if (out.get(t * c.qDim() + i).getFloat32() != 0.0f) {
                    nonZeroPadding++;
                }
            }
        }
        System.out.printf(
                "[T16] %s: %d outputs, worst %.4f of the bound, %d outside; wrong references"
                        + " outside: %s=%d, %s=%d, %s=%d, %s=%d of %d; nonzero padding rows %d%n",
                c,
                checked,
                worst,
                outside,
                wrongNames[0],
                wrongOutside[0],
                wrongNames[1],
                wrongOutside[1],
                wrongNames[2],
                wrongOutside[2],
                wrongNames[3],
                wrongOutside[3],
                wrongChecked,
                nonZeroPadding);
        List<String> failures = new ArrayList<>();
        if (outside > 0) {
            failures.add(c + ": " + outside + " of " + checked + " outputs outside the bound");
        }
        for (int w = 0; w < 4; w++) {
            if (w <= 1 && c.start == 0) {
                continue; // a first chunk's mask is aligned to it, and its keys are its prefix
            }
            if (w == 2 && c.kvMul() == 1 && c.kvHeads == 1) {
                continue;
            }
            if (2 * wrongOutside[w] <= wrongChecked) {
                failures.add(c + ": the reference '" + wrongNames[w] + "' is not told apart");
            }
        }
        if (nonZeroPadding > 0) {
            failures.add(c + ": " + nonZeroPadding + " outputs past the chunk's tokens not zero");
        }
        return failures;
    }

    /**
     * Softmax attention of {@code q} over positions {@code from..to} of the key/value head whose
     * columns start at {@code col}: {@code out[i] = sum p_j v_j[i]}, {@code mag[i] = sum p_j
     * |v_j[i]|}.
     */
    private static void attend(
            float[] q,
            float[][] k,
            float[][] v,
            int col,
            int from,
            int to,
            double scale,
            double[] out,
            double[] mag) {
        int d = q.length;
        int n = to - from + 1;
        double[] s = new double[n];
        double max = Double.NEGATIVE_INFINITY;
        for (int j = 0; j < n; j++) {
            double dot = 0;
            for (int i = 0; i < d; i++) {
                dot += (double) q[i] * k[from + j][col + i];
            }
            s[j] = dot * scale;
            max = Math.max(max, s[j]);
        }
        double sum = 0;
        for (int j = 0; j < n; j++) {
            s[j] = Math.exp(s[j] - max);
            sum += s[j];
        }
        java.util.Arrays.fill(out, 0);
        java.util.Arrays.fill(mag, 0);
        for (int j = 0; j < n; j++) {
            double p = s[j] / sum;
            for (int i = 0; i < d; i++) {
                double x = v[from + j][col + i];
                out[i] += p * x;
                mag[i] += p * Math.abs(x);
            }
        }
    }
}
