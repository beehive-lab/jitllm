package org.beehive.jitllm.backend.cpu;

import org.beehive.jitllm.auxiliary.Parallel;
import org.beehive.jitllm.inference.op.CpuOperations;
import org.beehive.jitllm.inference.state.KdaHostState;
import org.beehive.jitllm.inference.weights.KdaWeights;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * Kimi delta attention (KDA) on the host, one token at a time, for every family with KDA layers.
 *
 * <p>The families differ only in how the per-channel log decay is formed from {@code f_b(f_a(x)) +
 * dt_bias}; {@link Decay} names the two forms.
 */
public final class KdaForward {

    private KdaForward() {}

    /** How a KDA layer turns {@code z = f_b(f_a(x)) + dt_bias} into a log decay per channel. */
    public enum Decay {
        /**
         * {@code g = lowerBound * sigmoid(-a[h] * z)} (GLM-5.3-Flash): the decay {@code exp(g)}
         * stays within {@code (exp(lowerBound), 1)}.
         */
        BOUNDED_SIGMOID,
        /** {@code g = a[h] * softplus(z)} (Kimi-Linear), {@code a = -exp(A_log)}. */
        SOFTPLUS
    }

    /** The geometry and constants of one family's KDA layers. */
    public record Shape(
            int dim,
            int heads,
            int headDim,
            int convKernel,
            Decay decay,
            float lowerBound,
            float l2NormEps,
            float outNormEps) {

        int width() {
            return heads * headDim;
        }
    }

    // @formatter:off
    /**
     * KDA for one token of layer {@code l}: reads the normalized input {@code x}, writes {@code
     * attn_output}'s result into {@code result}.
     *
     * <ol>
     *   <li>query, key and value projections, each through its own causal depthwise convolution and
     *       a SiLU; query and key L2-normalized per head as {@code x / sqrt(sum(x²) + eps)}, the
     *       query scaled by {@code 1/sqrt(headDim)};
     *   <li>a per-channel log decay {@code g} ({@link Decay}) and a per-head {@code beta =
     *       sigmoid(b x)};
     *   <li>per head {@code S = diag(exp(g)) S}, {@code d = (v - Sᵀk) * beta}, {@code S += k ⊗ d},
     *       {@code o = Sᵀq};
     *   <li>{@code o} RMS-normalized per head and gated by {@code sigmoid(g_b(g_a(x)))}.
     * </ol>
     */
    // @formatter:on
    public static void forward(
            Shape shape,
            KdaWeights<FloatTensor> w,
            KdaHostState s,
            FloatTensor x,
            FloatTensor result,
            int l) {
        final int dim = shape.dim();
        final int heads = shape.heads();
        final int hd = shape.headDim();
        final int width = shape.width();
        final int kernel = shape.convKernel();

        CpuOperations.matVec(w.q()[l], x, s.q, width, dim);
        CpuOperations.matVec(w.k()[l], x, s.k, width, dim);
        CpuOperations.matVec(w.v()[l], x, s.v, width, dim);
        CpuOperations.causalConv1d(s.q, w.convQ()[l], s.convWindowQ[l], s.convQ, width, kernel);
        CpuOperations.causalConv1d(s.k, w.convK()[l], s.convWindowK[l], s.convK, width, kernel);
        CpuOperations.causalConv1d(s.v, w.convV()[l], s.convWindowV[l], s.convV, width, kernel);
        FloatTensor q = s.convQ;
        FloatTensor k = s.convK;
        FloatTensor v = s.convV;
        q.mapInPlace(KdaForward::silu);
        k.mapInPlace(KdaForward::silu);
        v.mapInPlace(KdaForward::silu);
        final float qScale = (float) (1.0 / Math.sqrt(hd));
        for (int h = 0; h < heads; h++) {
            l2Norm(q, h * hd, hd, qScale, shape.l2NormEps());
            l2Norm(k, h * hd, hd, 1f, shape.l2NormEps());
        }

        CpuOperations.matVec(w.decayA()[l], x, s.lowRank, hd, dim);
        CpuOperations.matVec(w.decayB()[l], s.lowRank, s.decay, width, hd);
        FloatTensor dtBias = w.dtBias()[l];
        FloatTensor a = w.a()[l];
        for (int c = 0; c < width; c++) {
            float rate = a.getFloat(c / hd);
            float z = s.decay.getFloat(c) + dtBias.getFloat(c);
            float g =
                    switch (shape.decay()) {
                        case BOUNDED_SIGMOID -> shape.lowerBound() * sigmoid(-rate * z);
                        case SOFTPLUS -> rate * softplus(z);
                    };
            s.decay.setFloat(c, (float) Math.exp(g));
        }
        CpuOperations.matVec(w.beta()[l], x, s.beta, heads, dim);
        for (int h = 0; h < heads; h++) {
            s.beta.setFloat(h, sigmoid(s.beta.getFloat(h)));
        }

        FloatTensor state = s.deltaState[l];
        FloatTensor decay = s.decay;
        FloatTensor out = s.out;
        FloatTensor beta = s.beta;
        Parallel.parallelFor(
                0, heads, h -> deltaRuleHead(q, k, v, decay, beta.getFloat(h), state, out, h, hd));

        CpuOperations.matVec(w.gateA()[l], x, s.lowRank, hd, dim);
        CpuOperations.matVec(w.gateB()[l], s.lowRank, s.gate, width, hd);
        FloatTensor outNorm = w.outNorm()[l];
        final float eps = shape.outNormEps();
        for (int h = 0; h < heads; h++) {
            int base = h * hd;
            float ss = out.reduce(base, hd, 0f, (acc, e) -> acc + e * e);
            float inv = (float) (1.0 / Math.sqrt(ss / hd + eps));
            for (int i = 0; i < hd; i++) {
                float g = sigmoid(s.gate.getFloat(base + i));
                out.setFloat(base + i, outNorm.getFloat(i) * inv * out.getFloat(base + i) * g);
            }
        }
        CpuOperations.matVec(w.output()[l], out, result, dim, width);
    }

    /** One head's delta-rule step with a decay per key channel; state row = key channel. */
    static void deltaRuleHead(
            FloatTensor q,
            FloatTensor k,
            FloatTensor v,
            FloatTensor decay,
            float beta,
            FloatTensor state,
            FloatTensor out,
            int h,
            int hd) {
        final int stateBase = h * hd * hd;
        final int base = h * hd;
        final float[] correction = new float[hd];
        for (int i = 0; i < hd; i++) {
            final float g = decay.getFloat(base + i);
            final float ki = k.getFloat(base + i);
            final int row = stateBase + i * hd;
            for (int j = 0; j < hd; j++) {
                float decayed = state.getFloat(row + j) * g;
                state.setFloat(row + j, decayed);
                correction[j] += decayed * ki;
            }
        }
        for (int j = 0; j < hd; j++) {
            correction[j] = (v.getFloat(base + j) - correction[j]) * beta;
        }
        final float[] o = new float[hd];
        for (int i = 0; i < hd; i++) {
            final float ki = k.getFloat(base + i);
            final float qi = q.getFloat(base + i);
            final int row = stateBase + i * hd;
            for (int j = 0; j < hd; j++) {
                float updated = state.getFloat(row + j) + ki * correction[j];
                state.setFloat(row + j, updated);
                o[j] += updated * qi;
            }
        }
        for (int j = 0; j < hd; j++) {
            out.setFloat(base + j, o[j]);
        }
    }

    /** {@code x / sqrt(sum(x²) + eps) * scale}: the norm FLA and llama.cpp use. */
    static void l2Norm(FloatTensor values, int offset, int size, float scale, float eps) {
        float ss = values.reduce(offset, size, 0f, (acc, e) -> acc + e * e);
        final float inv = (float) (scale / Math.sqrt(ss + eps));
        for (int i = 0; i < size; i++) {
            values.setFloat(offset + i, values.getFloat(offset + i) * inv);
        }
    }

    /** {@code log(1 + exp(x))}, linear above 20 as ggml's softplus is. */
    static float softplus(float x) {
        return x > 20f ? x : (float) Math.log1p(Math.exp(x));
    }

    static float silu(float x) {
        return x / (1f + (float) Math.exp(-x));
    }

    static float sigmoid(float x) {
        return 1f / (1f + (float) Math.exp(-x));
    }
}
