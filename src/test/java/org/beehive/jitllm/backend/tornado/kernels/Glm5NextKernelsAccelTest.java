package org.beehive.jitllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The {@code glm5next} decode kernels, each against a host evaluation in double precision: the
 * hyper-connection, several KDA steps, the indexer with sparse latent attention over a sequence
 * long enough for the selection to drop pools, and the router over 288 experts.
 */
public class Glm5NextKernelsAccelTest {

    private static FloatArray gaussian(int n, Random rng, float scale) {
        FloatArray a = new FloatArray(n);
        fill(a, rng, scale);
        return a;
    }

    private static void fill(FloatArray a, Random rng, float scale) {
        for (int i = 0; i < a.getSize(); i++) {
            a.set(i, (float) rng.nextGaussian() * scale);
        }
    }

    private static WorkerGrid lanes(int n, int local) {
        WorkerGrid g = new WorkerGrid1D((n + local - 1) / local * local);
        g.setLocalWork(local, 1, 1);
        return g;
    }

    private static WorkerGrid groups(int count, int local) {
        WorkerGrid g = new WorkerGrid1D(count * local);
        g.setLocalWork(local, 1, 1);
        return g;
    }

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private static void assertClose(String what, double expected, float actual, double tolerance) {
        assertEquals(what, expected, actual, tolerance * Math.max(1.0, Math.abs(expected)));
    }

    /**
     * The decode and the batched prefill compute the hyper-connection coefficients in the same
     * order, so a decode continuing a batched prefill sees the same values bit for bit.
     */
    @Test
    public void theDecodeAndBatchedHyperConnectionCoefficientsAgreeBitwise() throws Exception {
        int n = 4, dim = 4096, width = n * dim, mixCount = (2 + n) * n, iterations = 20;
        int chunks = Glm5NextKernels.HC_CHUNKS;
        float eps = 1e-6f, rmsEps = 1e-5f, postScale = 2f;
        Random rng = new Random(3);
        FloatArray streams = gaussian(width, rng, 1f);
        FloatArray fn = gaussian(mixCount * width, rng, 1f / (float) Math.sqrt(width));
        FloatArray base = gaussian(mixCount, rng, 0.5f);
        FloatArray scale = gaussian(3, rng, 0.2f);
        for (int i = 0; i < 3; i++) {
            scale.set(i, scale.get(i) + 1f);
        }
        FloatArray partials = new FloatArray(2 * mixCount * chunks);
        FloatArray mixes = new FloatArray(mixCount);
        FloatArray coeffs = new FloatArray(mixCount);
        FloatArray batchPartials = new FloatArray(2 * mixCount * chunks);
        FloatArray batchMixes = new FloatArray(mixCount);
        FloatArray batchCoeffs = new FloatArray(mixCount);

        TaskGraph g =
                new TaskGraph("hcb")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, streams, fn, base, scale)
                        .task(
                                "mix",
                                Glm5NextKernels::hcMixes,
                                new KernelContext(),
                                streams,
                                fn,
                                partials,
                                width,
                                mixCount)
                        .task(
                                "coef",
                                Glm5NextKernels::hcCoefficients,
                                new KernelContext(),
                                partials,
                                base,
                                scale,
                                mixes,
                                coeffs,
                                n,
                                width,
                                iterations,
                                eps,
                                postScale,
                                rmsEps)
                        .task(
                                "bmix",
                                Glm5NextBatchKernels::hcMixes,
                                new KernelContext(),
                                streams,
                                fn,
                                batchPartials,
                                width,
                                mixCount,
                                mixCount * chunks,
                                1)
                        .task(
                                "bfinish",
                                Glm5NextBatchKernels::hcFinishMixes,
                                new KernelContext(),
                                batchPartials,
                                batchMixes,
                                width,
                                mixCount,
                                mixCount * chunks,
                                rmsEps,
                                1)
                        .task(
                                "bcoef",
                                Glm5NextBatchKernels::hcCoefficients,
                                new KernelContext(),
                                batchMixes,
                                base,
                                scale,
                                batchCoeffs,
                                n,
                                iterations,
                                eps,
                                postScale,
                                1)
                        .transferToHost(
                                DataTransferMode.EVERY_EXECUTION,
                                partials,
                                mixes,
                                coeffs,
                                batchPartials,
                                batchMixes,
                                batchCoeffs);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("hcb.mix", groups(mixCount * chunks, 256));
        s.addWorkerGrid("hcb.coef", groups(1, 32));
        s.addWorkerGrid("hcb.bmix", groups(mixCount * chunks, 256));
        s.addWorkerGrid("hcb.bfinish", lanes(mixCount, 128));
        s.addWorkerGrid("hcb.bcoef", lanes(1, 32));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        for (int i = 0; i < 2 * mixCount * chunks; i++) {
            assertEquals("partial " + i, batchPartials.get(i), partials.get(i), 0f);
        }
        for (int k = 0; k < mixCount; k++) {
            assertEquals("mix " + k, batchMixes.get(k), mixes.get(k), 0f);
        }
        for (int k = 0; k < mixCount; k++) {
            assertEquals("coefficient " + k, batchCoeffs.get(k), coeffs.get(k), 0f);
        }
    }

    @Test
    public void theHyperConnectionMatchesTheHost() throws Exception {
        int n = 4, dim = 256, width = n * dim, mixCount = (2 + n) * n, iterations = 20;
        float eps = 1e-6f, rmsEps = 1e-5f, postScale = 2f;
        Random rng = new Random(1);
        FloatArray streams = gaussian(width, rng, 1f);
        FloatArray fn = gaussian(mixCount * width, rng, 1f / (float) Math.sqrt(width));
        FloatArray base = gaussian(mixCount, rng, 0.5f);
        FloatArray scale = gaussian(3, rng, 0.2f);
        for (int i = 0; i < 3; i++) {
            scale.set(i, scale.get(i) + 1f);
        }
        FloatArray branch = gaussian(dim, rng, 1f);
        float[] before = streams.toHeapArray();
        FloatArray mixes = new FloatArray(mixCount);
        FloatArray partials = new FloatArray(2 * mixCount * Glm5NextKernels.HC_CHUNKS);
        FloatArray coeffs = new FloatArray(mixCount);
        FloatArray residual = new FloatArray(width);
        FloatArray x = new FloatArray(dim);

        TaskGraph g =
                new TaskGraph("hc")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, streams, fn, base, scale, branch)
                        .task(
                                "mix",
                                Glm5NextKernels::hcMixes,
                                new KernelContext(),
                                streams,
                                fn,
                                partials,
                                width,
                                mixCount)
                        .task(
                                "coef",
                                Glm5NextKernels::hcCoefficients,
                                new KernelContext(),
                                partials,
                                base,
                                scale,
                                mixes,
                                coeffs,
                                n,
                                width,
                                iterations,
                                eps,
                                postScale,
                                rmsEps)
                        .task(
                                "collapse",
                                Glm5NextKernels::hcCollapse,
                                new KernelContext(),
                                streams,
                                coeffs,
                                residual,
                                x,
                                dim,
                                n)
                        .task(
                                "expand",
                                Glm5NextKernels::hcExpand,
                                new KernelContext(),
                                residual,
                                branch,
                                coeffs,
                                streams,
                                dim,
                                n)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, coeffs, x, streams);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("hc.mix", groups(mixCount * Glm5NextKernels.HC_CHUNKS, 256));
        s.addWorkerGrid("hc.coef", groups(1, 32));
        s.addWorkerGrid("hc.collapse", lanes(dim, 128));
        s.addWorkerGrid("hc.expand", lanes(width, 128));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(s).execute();
        }

        double ss = 0;
        for (float v : before) {
            ss += v * v;
        }
        double inv = 1.0 / Math.sqrt(ss / width + rmsEps);
        double[] mix = new double[mixCount];
        for (int k = 0; k < mixCount; k++) {
            double d = 0;
            for (int i = 0; i < width; i++) {
                d += fn.get(k * width + i) * before[i];
            }
            mix[k] = d * inv;
        }
        double[] pre = new double[n], post = new double[n], comb = new double[n * n];
        for (int i = 0; i < n; i++) {
            pre[i] = sigmoid(mix[i] * scale.get(0) + base.get(i)) + eps;
            post[i] = postScale * sigmoid(mix[n + i] * scale.get(1) + base.get(n + i));
        }
        for (int r = 0; r < n; r++) {
            double max = Double.NEGATIVE_INFINITY, sum = 0;
            for (int c = 0; c < n; c++) {
                comb[r * n + c] =
                        mix[2 * n + r * n + c] * scale.get(2) + base.get(2 * n + r * n + c);
                max = Math.max(max, comb[r * n + c]);
            }
            for (int c = 0; c < n; c++) {
                comb[r * n + c] = Math.exp(comb[r * n + c] - max);
                sum += comb[r * n + c];
            }
            for (int c = 0; c < n; c++) {
                comb[r * n + c] = comb[r * n + c] / sum + eps;
            }
        }
        for (int it = 0; it < iterations; it++) {
            if (it > 0) {
                for (int r = 0; r < n; r++) {
                    double sum = 0;
                    for (int c = 0; c < n; c++) {
                        sum += comb[r * n + c];
                    }
                    for (int c = 0; c < n; c++) {
                        comb[r * n + c] /= sum + eps;
                    }
                }
            }
            for (int c = 0; c < n; c++) {
                double sum = 0;
                for (int r = 0; r < n; r++) {
                    sum += comb[r * n + c];
                }
                for (int r = 0; r < n; r++) {
                    comb[r * n + c] /= sum + eps;
                }
            }
        }
        for (int i = 0; i < n; i++) {
            assertClose("pre " + i, pre[i], coeffs.get(i), 1e-4);
            assertClose("post " + i, post[i], coeffs.get(n + i), 1e-4);
        }
        for (int i = 0; i < n * n; i++) {
            assertClose("comb " + i, comb[i], coeffs.get(2 * n + i), 1e-4);
        }
        for (int i = 0; i < dim; i++) {
            double collapsed = 0;
            for (int m = 0; m < n; m++) {
                collapsed += pre[m] * before[m * dim + i];
            }
            assertClose("x " + i, collapsed, x.get(i), 1e-4);
            for (int k = 0; k < n; k++) {
                double v = post[k] * branch.get(i);
                for (int m = 0; m < n; m++) {
                    v += comb[m * n + k] * before[m * dim + i];
                }
                assertClose("stream " + k + "," + i, v, streams.get(k * dim + i), 1e-4);
            }
        }
    }

    @Test
    public void kdaStepsMatchTheHost() throws Exception {
        int heads = 4, hd = 32, width = heads * hd, steps = 6;
        float lower = -5f, eps = 1e-5f, l2Eps = 1e-6f;
        Random rng = new Random(2);
        FloatArray q = new FloatArray(width), k = new FloatArray(width), v = new FloatArray(width);
        FloatArray decay = new FloatArray(width), beta = new FloatArray(heads);
        FloatArray gate = new FloatArray(width);
        FloatArray dtBias = gaussian(width, rng, 0.5f);
        FloatArray aLog = gaussian(heads, rng, 0.5f);
        FloatArray weight = gaussian(hd, rng, 0.1f);
        for (int i = 0; i < hd; i++) {
            weight.set(i, weight.get(i) + 1f);
        }
        FloatArray state = new FloatArray(heads * hd * hd);
        FloatArray out = new FloatArray(width);
        double[] host = new double[heads * hd * hd];

        TaskGraph g =
                new TaskGraph("kda")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                q,
                                k,
                                v,
                                decay,
                                beta,
                                gate,
                                dtBias,
                                aLog,
                                weight)
                        .transferToDevice(DataTransferMode.FIRST_EXECUTION, state)
                        .task(
                                "decay",
                                Glm5NextKernels::kdaDecayAndBeta,
                                new KernelContext(),
                                decay,
                                dtBias,
                                aLog,
                                beta,
                                width,
                                hd,
                                heads,
                                lower)
                        .task(
                                "nq",
                                Glm5NextKernels::kdaL2Norm,
                                new KernelContext(),
                                q,
                                heads,
                                hd,
                                (float) (1.0 / Math.sqrt(hd)),
                                l2Eps)
                        .task(
                                "nk",
                                Glm5NextKernels::kdaL2Norm,
                                new KernelContext(),
                                k,
                                heads,
                                hd,
                                1f,
                                l2Eps)
                        .task(
                                "delta",
                                Glm5NextKernels::kdaDeltaRule,
                                new KernelContext(),
                                q,
                                k,
                                v,
                                decay,
                                beta,
                                state,
                                out,
                                heads,
                                hd,
                                0)
                        .task(
                                "norm",
                                Glm5NextKernels::kdaGatedNorm,
                                new KernelContext(),
                                out,
                                gate,
                                weight,
                                heads,
                                hd,
                                eps)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("kda.decay", lanes(width, 64));
        s.addWorkerGrid("kda.nq", groups(heads, 32));
        s.addWorkerGrid("kda.nk", groups(heads, 32));
        s.addWorkerGrid(
                "kda.delta",
                groups(
                        heads
                                * ((hd + Glm5NextKernels.DELTA_COLUMNS - 1)
                                        / Glm5NextKernels.DELTA_COLUMNS),
                        Glm5NextKernels.DELTA_COLUMNS * Glm5NextKernels.DELTA_CHUNKS));
        s.addWorkerGrid("kda.norm", groups(heads, 32));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            for (int step = 0; step < steps; step++) {
                fill(q, rng, 1f);
                fill(k, rng, 1f);
                fill(v, rng, 1f);
                fill(decay, rng, 1f);
                fill(beta, rng, 1f);
                fill(gate, rng, 1f);
                double[] hq = new double[width], hk = new double[width], hv = new double[width];
                double[] hDecay = new double[width],
                        hBeta = new double[heads],
                        hGate = new double[width];
                for (int i = 0; i < width; i++) {
                    hq[i] = q.get(i);
                    hk[i] = k.get(i);
                    hv[i] = v.get(i);
                    hDecay[i] =
                            Math.exp(
                                    lower
                                            * sigmoid(
                                                    Math.exp(aLog.get(i / hd))
                                                            * (decay.get(i) + dtBias.get(i))));
                    hGate[i] = gate.get(i);
                }
                for (int h = 0; h < heads; h++) {
                    hBeta[h] = sigmoid(beta.get(h));
                }
                plan.withGridScheduler(s).execute();

                for (int h = 0; h < heads; h++) {
                    double sq = 0, sk = 0;
                    for (int i = 0; i < hd; i++) {
                        sq += hq[h * hd + i] * hq[h * hd + i];
                        sk += hk[h * hd + i] * hk[h * hd + i];
                    }
                    for (int i = 0; i < hd; i++) {
                        hq[h * hd + i] /= Math.sqrt(sq + l2Eps) * Math.sqrt(hd);
                        hk[h * hd + i] /= Math.sqrt(sk + l2Eps);
                    }
                    double[] o = new double[hd];
                    for (int j = 0; j < hd; j++) {
                        double corr = 0;
                        for (int i = 0; i < hd; i++) {
                            int e = (h * hd + i) * hd + j;
                            host[e] *= hDecay[h * hd + i];
                            corr += host[e] * hk[h * hd + i];
                        }
                        double d = (hv[h * hd + j] - corr) * hBeta[h];
                        for (int i = 0; i < hd; i++) {
                            int e = (h * hd + i) * hd + j;
                            host[e] += hk[h * hd + i] * d;
                            o[j] += host[e] * hq[h * hd + i];
                        }
                    }
                    double ss = 0;
                    for (int j = 0; j < hd; j++) {
                        ss += o[j] * o[j];
                    }
                    double inv = 1.0 / Math.sqrt(ss / hd + eps);
                    for (int j = 0; j < hd; j++) {
                        double expected = weight.get(j) * inv * o[j] * sigmoid(hGate[h * hd + j]);
                        assertClose(
                                "step " + step + " out " + h + "," + j,
                                expected,
                                out.get(h * hd + j),
                                1e-3);
                    }
                }
            }
        }
    }

    @Test
    public void theIndexerAndSparseAttentionMatchTheHost() throws Exception {
        int ih = 2, id = 32, pool = 4, budget = 3, heads = 2, rank = 32, positions = 30;
        int maxPools = positions / pool + 1;
        float eps = 1e-6f, scale = 0.25f;
        Random rng = new Random(3);
        FloatArray rawKey = new FloatArray(id), gateRow = new FloatArray(id);
        FloatArray normWeight = gaussian(id, rng, 0.1f), normBias = gaussian(id, rng, 0.1f);
        for (int i = 0; i < id; i++) {
            normWeight.set(i, normWeight.get(i) + 1f);
        }
        FloatArray ape = gaussian(pool * id, rng, 0.5f);
        FloatArray keyCache = new FloatArray(positions * id);
        FloatArray gateCache = new FloatArray(positions * id);
        FloatArray poolKeys = new FloatArray(maxPools * id);
        FloatArray query = new FloatArray(ih * id), headWeights = new FloatArray(ih);
        FloatArray scores = new FloatArray(maxPools);
        IntArray selected = new IntArray(maxPools);
        FloatArray latents = gaussian(positions * rank, rng, 1f);
        FloatArray absorbed = new FloatArray(heads * rank);
        FloatArray scratch = new FloatArray(heads * positions);
        FloatArray latentOut = new FloatArray(heads * rank);
        IntArray positionHolder = new IntArray(1);

        TaskGraph g =
                new TaskGraph("idx")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                rawKey,
                                gateRow,
                                query,
                                headWeights,
                                absorbed,
                                positionHolder)
                        .transferToDevice(
                                DataTransferMode.FIRST_EXECUTION,
                                normWeight,
                                normBias,
                                ape,
                                keyCache,
                                gateCache,
                                poolKeys,
                                latents)
                        .task(
                                "store",
                                Glm5NextKernels::indexerStore,
                                new KernelContext(),
                                rawKey,
                                gateRow,
                                normWeight,
                                normBias,
                                keyCache,
                                gateCache,
                                positionHolder,
                                id,
                                0,
                                eps)
                        .task(
                                "pool",
                                Glm5NextKernels::indexerPoolKey,
                                new KernelContext(),
                                keyCache,
                                gateCache,
                                ape,
                                poolKeys,
                                positionHolder,
                                id,
                                pool,
                                0,
                                0)
                        .task(
                                "score",
                                Glm5NextKernels::indexerScores,
                                new KernelContext(),
                                query,
                                headWeights,
                                poolKeys,
                                scores,
                                positionHolder,
                                ih,
                                id,
                                pool,
                                maxPools,
                                0)
                        .task(
                                "select",
                                Glm5NextKernels::indexerSelect,
                                new KernelContext(),
                                scores,
                                selected,
                                positionHolder,
                                pool,
                                maxPools,
                                budget)
                        .task(
                                "attend",
                                Glm5NextKernels::sparseLatentAttention,
                                new KernelContext(),
                                absorbed,
                                latents,
                                selected,
                                scratch,
                                latentOut,
                                positionHolder,
                                heads,
                                rank,
                                pool,
                                positions,
                                0,
                                1,
                                scale)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, selected, latentOut);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("idx.store", groups(1, 32));
        s.addWorkerGrid("idx.pool", lanes(id, 32));
        s.addWorkerGrid("idx.score", lanes(maxPools * Glm5NextKernels.INDEX_LANES, 64));
        s.addWorkerGrid("idx.select", groups(1, 64));
        s.addWorkerGrid("idx.attend", groups(heads, 64));

        double[][] keys = new double[positions][id];
        double[][] gates = new double[positions][id];
        double[][] pooled = new double[maxPools][id];
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            for (int t = 0; t < positions; t++) {
                fill(rawKey, rng, 1f);
                fill(gateRow, rng, 1f);
                fill(query, rng, 1f);
                fill(headWeights, rng, 1f);
                fill(absorbed, rng, 1f);
                positionHolder.set(0, t);
                double mean = 0, var = 0;
                for (int c = 0; c < id; c++) {
                    mean += rawKey.get(c);
                }
                mean /= id;
                for (int c = 0; c < id; c++) {
                    var += (rawKey.get(c) - mean) * (rawKey.get(c) - mean);
                }
                var /= id;
                for (int c = 0; c < id; c++) {
                    keys[t][c] =
                            (rawKey.get(c) - mean) / Math.sqrt(var + eps) * normWeight.get(c)
                                    + normBias.get(c);
                    gates[t][c] = gateRow.get(c);
                }
                if ((t + 1) % pool == 0) {
                    int p = t / pool;
                    for (int c = 0; c < id; c++) {
                        double max = Double.NEGATIVE_INFINITY, sum = 0, acc = 0;
                        for (int j = 0; j < pool; j++) {
                            max = Math.max(max, gates[p * pool + j][c] + ape.get(j * id + c));
                        }
                        for (int j = 0; j < pool; j++) {
                            double e = Math.exp(gates[p * pool + j][c] + ape.get(j * id + c) - max);
                            sum += e;
                            acc += e * keys[p * pool + j][c];
                        }
                        pooled[p][c] = acc / sum;
                    }
                }
                plan.withGridScheduler(s).execute();

                int complete = (t + 1) / pool;
                List<Integer> order = new ArrayList<>();
                double[] score = new double[complete];
                for (int p = 0; p < complete; p++) {
                    order.add(p);
                    for (int h = 0; h < ih; h++) {
                        double d = 0;
                        for (int c = 0; c < id; c++) {
                            d += query.get(h * id + c) * pooled[p][c];
                        }
                        score[p] +=
                                headWeights.get(h) / Math.sqrt(ih) * Math.max(d / Math.sqrt(id), 0);
                    }
                }
                order.sort(
                        (a, b) ->
                                score[b] != score[a]
                                        ? Double.compare(score[b], score[a])
                                        : Integer.compare(a, b));
                boolean[] chosen = new boolean[maxPools];
                for (int i = 0; i < Math.min(budget, complete); i++) {
                    chosen[order.get(i)] = true;
                }
                for (int p = 0; p < maxPools; p++) {
                    assertEquals(
                            "position " + t + " pool " + p, chosen[p] ? 1 : 0, selected.get(p));
                }

                List<Integer> attended = new ArrayList<>();
                for (int u = 0; u <= t; u++) {
                    if (u >= complete * pool || chosen[u / pool]) {
                        attended.add(u);
                    }
                }
                for (int h = 0; h < heads; h++) {
                    double[] w = new double[attended.size()];
                    double max = Double.NEGATIVE_INFINITY, sum = 0;
                    for (int j = 0; j < w.length; j++) {
                        double d = 0;
                        for (int r = 0; r < rank; r++) {
                            d +=
                                    absorbed.get(h * rank + r)
                                            * latents.get(attended.get(j) * rank + r);
                        }
                        w[j] = d * scale;
                        max = Math.max(max, w[j]);
                    }
                    for (int j = 0; j < w.length; j++) {
                        w[j] = Math.exp(w[j] - max);
                        sum += w[j];
                    }
                    for (int r = 0; r < rank; r++) {
                        double acc = 0;
                        for (int j = 0; j < w.length; j++) {
                            acc += w[j] * latents.get(attended.get(j) * rank + r);
                        }
                        assertClose(
                                "position " + t + " head " + h + " latent " + r,
                                acc / sum,
                                latentOut.get(h * rank + r),
                                1e-4);
                    }
                }
            }
        }
    }

    @Test
    public void theRouterMatchesTheHost() throws Exception {
        int experts = 288, used = 8;
        float scale = 2.5f;
        Random rng = new Random(4);
        FloatArray logits = gaussian(experts, rng, 2f);
        FloatArray bias = gaussian(experts, rng, 0.1f);
        IntArray ids = new IntArray(used);
        FloatArray weights = new FloatArray(used);
        TaskGraph g =
                new TaskGraph("route")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, logits, bias)
                        .task(
                                "topk",
                                Glm5NextKernels::routerTopKSigmoid,
                                new KernelContext(),
                                logits,
                                bias,
                                ids,
                                weights,
                                experts,
                                used,
                                1,
                                scale)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, ids, weights);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("route.topk", groups(1, 128));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        double[] select = new double[experts];
        for (int e = 0; e < experts; e++) {
            select[e] = sigmoid(logits.get(e)) + bias.get(e);
        }
        Integer[] order = new Integer[experts];
        for (int e = 0; e < experts; e++) {
            order[e] = e;
        }
        Arrays.sort(order, (a, b) -> Double.compare(select[b], select[a]));
        double total = 0;
        for (int k = 0; k < used; k++) {
            total += sigmoid(logits.get(order[k]));
        }
        for (int k = 0; k < used; k++) {
            assertEquals("expert " + k, (int) order[k], ids.get(k));
            assertClose(
                    "weight " + k,
                    sigmoid(logits.get(order[k])) / total * scale,
                    weights.get(k),
                    1e-5);
        }
    }

    /** The batched router at the model's size: 288 experts, eight chosen, for every row. */
    @Test
    public void theBatchedRouterMatchesTheHost() throws Exception {
        int experts = 288, used = 8, batch = 64;
        float scale = 2.5f;
        Random rng = new Random(5);
        FloatArray logits = gaussian(batch * experts, rng, 2f);
        FloatArray bias = gaussian(experts, rng, 0.1f);
        IntArray ids = new IntArray(batch * used);
        FloatArray weights = new FloatArray(batch * used);
        TaskGraph g =
                new TaskGraph("routeb")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, logits, bias)
                        .task(
                                "topk",
                                Glm5NextBatchKernels::routerTopKSigmoid,
                                new KernelContext(),
                                logits,
                                bias,
                                ids,
                                weights,
                                experts,
                                used,
                                1,
                                scale,
                                batch)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, ids, weights);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("routeb.topk", groups(batch, Glm5NextBatchKernels.GROUP));
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(g.snapshot())) {
            plan.withGridScheduler(s).execute();
        }
        for (int b = 0; b < batch; b++) {
            final int row = b;
            double[] select = new double[experts];
            for (int e = 0; e < experts; e++) {
                select[e] = sigmoid(logits.get(row * experts + e)) + bias.get(e);
            }
            Integer[] order = new Integer[experts];
            for (int e = 0; e < experts; e++) {
                order[e] = e;
            }
            Arrays.sort(order, (x, y) -> Double.compare(select[y], select[x]));
            for (int k = 0; k < used; k++) {
                assertEquals("row " + b + " expert " + k, (int) order[k], ids.get(b * used + k));
            }
        }
    }
}
