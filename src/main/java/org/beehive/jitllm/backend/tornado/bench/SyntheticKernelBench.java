package org.beehive.jitllm.backend.tornado.bench;

import java.util.Random;
import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Synthetic, model-free kernel benchmarks that show <i>why</i> batched decode wins, by measuring
 * the two regimes separately against the same kernels the decode path uses.
 *
 * <p>Attention is memory-bound: each slot streams its own KV cache, so batching B slots into one
 * launch saves launch overhead but not bandwidth — the win is small. Projections are the opposite:
 * in batched form each weight row is loaded into shared memory once per workgroup and reused across
 * all B tokens (arithmetic intensity ~B, compute-bound), while the single-token path re-reads every
 * weight row per token (memory-bound). That asymmetry is the reason batching turns decode matvecs
 * into a compute-bound win.
 *
 * <p>Both use synthetic data at Llama-3.2-1B geometry and load no model, so they measure the
 * kernels rather than an end-to-end pipeline. They are reported through {@code JitllmBench} ({@code
 * --synthetic}); this class only measures and returns, so that console I/O and CLI types stay out
 * of the backend.
 *
 * <p>Lives in the backend package because it builds {@link TaskGraph} and {@link GridScheduler}
 * directly, which Rule 11 confines here.
 */
public final class SyntheticKernelBench {

    private SyntheticKernelBench() {}

    /** Llama-3.2-1B geometry, one layer — enough to exercise the kernels. */
    private static final int DIM = 2048;

    private static final int N_HEADS = 32;
    private static final int HEAD_SIZE = 64;
    private static final int N_KV_HEADS = 8;
    private static final int KV_DIM = N_KV_HEADS * HEAD_SIZE;
    private static final int KV_MUL = N_HEADS / N_KV_HEADS;
    private static final int N_LAYERS = 1;
    private static final int CTX = 1024;
    private static final int LAYER = 0;
    private static final int ATTN_LOCAL = HEAD_SIZE;
    private static final int PROJ_LOCAL = 128;

    /** Untimed executions before each timed loop. */
    private static final int WARMUP = 10;

    /**
     * One measurement: the same work batched across B slots versus one slot at a time.
     *
     * @param name which kernel pair was measured
     * @param shape the geometry, for the report
     * @param batchedMs milliseconds per batched step
     * @param singleMs milliseconds for a single slot
     * @param batch B — how many slots the batched step covered
     * @param maxRelError worst relative error against the CPU reference, or NaN if unchecked
     * @param outOfTol elements past tolerance, or -1 if unchecked
     * @param checked elements compared, or 0 if unchecked
     */
    public record Measurement(
            String name,
            String shape,
            double batchedMs,
            double singleMs,
            int batch,
            double maxRelError,
            int outOfTol,
            int checked) {

        /** What batching bought: B sequential launches collapsed into one. */
        public double speedup() {
            return (singleMs * batch) / batchedMs;
        }
    }

    /**
     * Batched decode attention: B independent sequences, each with its own KV cache and one query
     * token, each attending 0.pos of its own cache. Verified against a CPU reference.
     */
    public static Measurement decodeAttention(int batch, int seqLen, int iterations)
            throws TornadoExecutionPlanException {
        Random rnd = new Random(42);

        FloatArray q = new FloatArray(batch * DIM);
        FloatArray keyCache = new FloatArray(batch * N_LAYERS * CTX * KV_DIM);
        FloatArray valueCache = new FloatArray(batch * N_LAYERS * CTX * KV_DIM);
        FloatArray xb = new FloatArray(batch * DIM);
        IntArray seqPos = new IntArray(batch);

        for (int i = 0; i < batch * DIM; i++) {
            q.set(i, rnd.nextFloat() - 0.5f);
        }
        for (int b = 0; b < batch; b++) {
            seqPos.set(b, seqLen - 1);
            long base = (long) b * N_LAYERS * CTX * KV_DIM;
            for (int t = 0; t < seqLen; t++) {
                for (int d = 0; d < KV_DIM; d++) {
                    keyCache.set((int) (base + (long) t * KV_DIM + d), rnd.nextFloat() - 0.5f);
                    valueCache.set((int) (base + (long) t * KV_DIM + d), rnd.nextFloat() - 0.5f);
                }
            }
        }

        double batchedMs;
        double maxRel;
        int bad;
        try (TornadoExecutionPlan plan =
                attentionPlan(batch, q, keyCache, valueCache, xb, seqPos)) {
            warmUp(plan);

            float[] ref = attentionReference(q, keyCache, valueCache, seqPos, batch);
            maxRel = 0.0;
            bad = 0;
            for (int i = 0; i < batch * DIM; i++) {
                double rel = Math.abs(ref[i] - xb.get(i)) / Math.max(1e-3, Math.abs(ref[i]));
                maxRel = Math.max(maxRel, rel);
                if (rel > 2e-2) {
                    bad++;
                }
            }

            batchedMs = timeExecutions(plan, iterations);
        }

        double singleMs;
        Random one = new Random(7);
        FloatArray q1 = new FloatArray(DIM);
        FloatArray k1 = new FloatArray(N_LAYERS * CTX * KV_DIM);
        FloatArray v1 = new FloatArray(N_LAYERS * CTX * KV_DIM);
        FloatArray xb1 = new FloatArray(DIM);
        IntArray pos1 = new IntArray(1);
        pos1.set(0, seqLen - 1);
        for (int i = 0; i < DIM; i++) {
            q1.set(i, one.nextFloat() - 0.5f);
        }
        for (int i = 0; i < N_LAYERS * CTX * KV_DIM; i++) {
            k1.set(i, one.nextFloat() - 0.5f);
            v1.set(i, one.nextFloat() - 0.5f);
        }
        try (TornadoExecutionPlan plan = attentionPlan(1, q1, k1, v1, xb1, pos1)) {
            warmUp(plan);
            singleMs = timeExecutions(plan, iterations);
        }

        return new Measurement(
                "decode-attention",
                "B=" + batch + " seqLen=" + seqLen + " heads=" + N_HEADS + " headSize=" + HEAD_SIZE,
                batchedMs,
                singleMs,
                batch,
                maxRel,
                bad,
                batch * DIM);
    }

    private static TornadoExecutionPlan attentionPlan(
            int batch,
            FloatArray q,
            FloatArray keyCache,
            FloatArray valueCache,
            FloatArray xb,
            IntArray seqPos) {
        KernelContext context = new KernelContext();
        WorkerGrid1D worker = new WorkerGrid1D(batch * N_HEADS * ATTN_LOCAL);
        worker.setLocalWork(ATTN_LOCAL, 1, 1);
        String graph = "attn" + batch;
        TaskGraph tg =
                new TaskGraph(graph)
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, q, keyCache, valueCache, seqPos)
                        .task(
                                "attn",
                                TransformerBatchPrefillKernels::batchedDecodeAttention,
                                context,
                                seqPos,
                                q,
                                keyCache,
                                valueCache,
                                xb,
                                N_HEADS,
                                HEAD_SIZE,
                                KV_DIM,
                                KV_MUL,
                                LAYER,
                                N_LAYERS,
                                CTX,
                                DIM)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, xb);
        TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot());
        plan.withGridScheduler(new GridScheduler(graph + ".attn", worker));
        return plan;
    }

    /**
     * Projection C[B,N] = X[B,K] · Wᵀ, batched against single-token. The batched kernel loads each
     * weight row once per workgroup and reuses it across all B tokens; the single-token kernel
     * re-reads it per token.
     */
    public static Measurement projection(int batch, int k, int n, int iterations)
            throws TornadoExecutionPlanException {
        Random rnd = new Random(1);

        FloatArray w = new FloatArray(n * k);
        for (int i = 0; i < n * k; i++) {
            w.set(i, rnd.nextFloat() - 0.5f);
        }
        FloatArray xB = new FloatArray(batch * k);
        for (int i = 0; i < batch * k; i++) {
            xB.set(i, rnd.nextFloat() - 0.5f);
        }
        FloatArray cB = new FloatArray(batch * n);

        double batchedMs;
        double maxRel;
        int checked;
        KernelContext context = new KernelContext();
        WorkerGrid1D worker = new WorkerGrid1D(n * PROJ_LOCAL);
        worker.setLocalWork(PROJ_LOCAL, 1, 1);
        TaskGraph batchedGraph =
                new TaskGraph("proj")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, xB, w)
                        .task(
                                "proj",
                                SyntheticKernelBench::batchedProjection,
                                context,
                                xB,
                                w,
                                cB,
                                batch,
                                k,
                                n)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, cB);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(batchedGraph.snapshot())) {
            plan.withGridScheduler(new GridScheduler("proj.proj", worker));
            warmUp(plan);

            // Token 0 only: the reference is O(K) per column and the columns share the kernel.
            checked = Math.min(n, 256);
            maxRel = 0.0;
            for (int col = 0; col < checked; col++) {
                float ref = 0.0f;
                for (int i = 0; i < k; i++) {
                    ref += w.get(col * k + i) * xB.get(i);
                }
                maxRel =
                        Math.max(
                                maxRel,
                                Math.abs(ref - cB.get(col)) / Math.max(1e-2, Math.abs(ref)));
            }

            batchedMs = timeExecutions(plan, iterations);
        }

        FloatArray x1 = new FloatArray(k);
        for (int i = 0; i < k; i++) {
            x1.set(i, rnd.nextFloat() - 0.5f);
        }
        FloatArray c1 = new FloatArray(n);
        double singleMs;
        KernelContext singleContext = new KernelContext();
        WorkerGrid1D singleWorker = new WorkerGrid1D(n * PROJ_LOCAL);
        singleWorker.setLocalWork(PROJ_LOCAL, 1, 1);
        TaskGraph singleGraph =
                new TaskGraph("proj1")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, x1, w)
                        .task(
                                "proj",
                                SyntheticKernelBench::singleProjection,
                                singleContext,
                                x1,
                                w,
                                c1,
                                k,
                                n)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, c1);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(singleGraph.snapshot())) {
            plan.withGridScheduler(new GridScheduler("proj1.proj", singleWorker));
            warmUp(plan);
            singleMs = timeExecutions(plan, iterations);
        }

        return new Measurement(
                "projection",
                "B=" + batch + " K=" + k + " N=" + n,
                batchedMs,
                singleMs,
                batch,
                maxRel,
                -1,
                checked);
    }

    /** One workgroup per output column n: load W[n] to shared, apply to all B tokens. */
    public static void batchedProjection(
            KernelContext ctx, FloatArray x, FloatArray w, FloatArray c, int batch, int k, int n) {
        int col = ctx.groupIdx;
        int tid = ctx.localIdx;
        int localSize = ctx.localGroupSizeX;
        float[] wRow = ctx.allocateFloatLocalArray(2048); // K <= 2048
        for (int i = tid; i < k; i += localSize) {
            wRow[i] = w.get(col * k + i);
        }
        ctx.localBarrier();
        for (int b = tid; b < batch; b += localSize) {
            float acc = 0.0f;
            int xOffset = b * k;
            for (int i = 0; i < k; i++) {
                acc += wRow[i] * x.get(xOffset + i);
            }
            c.set(b * n + col, acc);
        }
    }

    /** Single-token matvec: one workgroup per output column, W[n] re-read from global. */
    public static void singleProjection(
            KernelContext ctx, FloatArray x, FloatArray w, FloatArray c, int k, int n) {
        int col = ctx.groupIdx;
        int tid = ctx.localIdx;
        int localSize = ctx.localGroupSizeX;
        float[] partial = ctx.allocateFloatLocalArray(PROJ_LOCAL);
        float acc = 0.0f;
        for (int i = tid; i < k; i += localSize) {
            acc += w.get(col * k + i) * x.get(i);
        }
        partial[tid] = acc;
        ctx.localBarrier();
        for (int s = localSize / 2; s > 0; s >>= 1) {
            if (tid < s) {
                partial[tid] += partial[tid + s];
            }
            ctx.localBarrier();
        }
        if (tid == 0) {
            c.set(col, partial[0]);
        }
    }

    private static void warmUp(TornadoExecutionPlan plan) throws TornadoExecutionPlanException {
        for (int i = 0; i < WARMUP; i++) {
            plan.execute();
        }
    }

    private static double timeExecutions(TornadoExecutionPlan plan, int iterations)
            throws TornadoExecutionPlanException {
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            plan.execute();
        }
        return (System.nanoTime() - start) / 1e6 / iterations;
    }

    private static float[] attentionReference(
            FloatArray q, FloatArray keyCache, FloatArray valueCache, IntArray seqPos, int batch) {
        float[] out = new float[batch * DIM];
        float scale = (float) (1.0 / Math.sqrt(HEAD_SIZE));
        for (int b = 0; b < batch; b++) {
            int pos = seqPos.get(b);
            long base = (long) b * N_LAYERS * CTX * KV_DIM;
            for (int h = 0; h < N_HEADS; h++) {
                int kvHead = h / KV_MUL;
                int qOffset = b * DIM + h * HEAD_SIZE;
                float[] scores = new float[pos + 1];
                float max = Float.NEGATIVE_INFINITY;
                for (int t = 0; t <= pos; t++) {
                    float s = 0.0f;
                    for (int d = 0; d < HEAD_SIZE; d++) {
                        s +=
                                q.get(qOffset + d)
                                        * keyCache.get(
                                                (int)
                                                        (base
                                                                + (long) t * KV_DIM
                                                                + kvHead * HEAD_SIZE
                                                                + d));
                    }
                    s *= scale;
                    scores[t] = s;
                    max = Math.max(max, s);
                }
                float sum = 0.0f;
                for (int t = 0; t <= pos; t++) {
                    scores[t] = (float) Math.exp(scores[t] - max);
                    sum += scores[t];
                }
                float inv = sum > 0 ? 1.0f / sum : 0.0f;
                for (int d = 0; d < HEAD_SIZE; d++) {
                    float acc = 0.0f;
                    for (int t = 0; t <= pos; t++) {
                        acc +=
                                scores[t]
                                        * inv
                                        * valueCache.get(
                                                (int)
                                                        (base
                                                                + (long) t * KV_DIM
                                                                + kvHead * HEAD_SIZE
                                                                + d));
                    }
                    out[qOffset + d] = acc;
                }
            }
        }
        return out;
    }
}
