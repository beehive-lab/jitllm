package org.beehive.jllm.golden;

import java.nio.file.Path;
import org.beehive.jllm.inference.weights.standard.StandardWeights;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.granite.GraniteConfiguration;
import org.beehive.jllm.model.loader.ModelLoader;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tensor.standard.FloatTensor;
import uk.ac.manchester.tornado.api.types.HalfFloat;

/**
 * Diagnostic (CPU only): how much precision the FP16 activation format costs a given model.
 *
 * <p>The FP16 GPU path stores the normalized activation as FP16 before every matmul; the Q8_0 path
 * keeps FP32. That is why the FP16 configurations sit ~1e-3 from the CPU while the Q8_0 ones sit
 * ~1e-5. This measures the format's cost directly — round layer 0's normalized activation to FP16
 * and compare — so a model whose CPU↔GPU gap is larger than its peers' can be checked against what
 * its own numbers predict, instead of being assumed either fine or broken.
 *
 * <p>Also reports the magnitude range, since FP16 overflows above 65504 and loses relative
 * precision below ~6e-5 — the plausible ways a model with large activation multipliers (Granite
 * scales its embeddings by 12) could be hurt more than the format's nominal ~5e-4.
 */
public final class Fp16Sensitivity {

    public static void main(String[] args) throws Exception {
        for (String spec : System.getProperty("fp16.models").split(",")) {
            report(Path.of(spec));
        }
    }

    private static void report(Path modelPath) throws Exception {
        Model model = ModelLoader.loadModel(modelPath, 512, true, false);
        Configuration config = model.configuration();
        StandardWeights w = (StandardWeights) model.weights();
        int dim = config.dim();
        float eps = config.rmsNormEps();
        float embeddingScale = config instanceof GraniteConfiguration g ? g.embeddingScale() : 1.0f;

        // A spread of token ids rather than one, so the answer is not a property of a single row.
        int[] tokens = {1, 100, 1000, 5000, 12345};
        double worstRel = 0;
        double sumRel = 0;
        double maxAbsX = 0;
        double minNonZeroAbsX = Double.MAX_VALUE;
        int overflow = 0;
        int subnormal = 0;

        for (int token : tokens) {
            float[] x = new float[dim];
            w.token_embedding_table.copyTo(token * dim, new ArrayFloatTensor(x), 0, dim);
            for (int i = 0; i < dim; i++) {
                x[i] *= embeddingScale;
            }

            double ss = 0;
            for (float v : x) {
                ss += (double) v * v;
            }
            float scale = (float) (1.0 / Math.sqrt(ss / dim + eps));

            double sqErr = 0;
            double sqRef = 0;
            for (int i = 0; i < dim; i++) {
                float xb = w.rms_att_weight[0].getFloat(i) * (scale * x[i]);
                float rounded = new HalfFloat(xb).getFloat32();
                double d = (double) xb - rounded;
                sqErr += d * d;
                sqRef += (double) xb * xb;
                double a = Math.abs(xb);
                maxAbsX = Math.max(maxAbsX, a);
                if (a > 0) {
                    minNonZeroAbsX = Math.min(minNonZeroAbsX, a);
                }
                if (a > 65504.0) {
                    overflow++;
                }
                if (a > 0 && a < 6.1e-5) {
                    subnormal++;
                }
            }
            double rel = Math.sqrt(sqErr / sqRef);
            worstRel = Math.max(worstRel, rel);
            sumRel += rel;
        }

        System.out.printf(
                "%-34s embScale=%-5.3g  fp16 round-trip relL2: mean=%.3e worst=%.3e"
                        + "  |xb| in [%.3g, %.3g]  overflow=%d subnormal=%d%n",
                modelPath.getFileName(),
                embeddingScale,
                sumRel / tokens.length,
                worstRel,
                minNonZeroAbsX,
                maxAbsX,
                overflow,
                subnormal);

        // Denormal FP16 weights (|w| < 2^-14) matter because FP16FloatTensor.vectorDot converts
        // with an explicit denormals-are-zero shortcut, so the CPU drops them while the GPU
        // converts them properly. That makes the CPU reference — not the GPU — the approximate
        // side, by an amount that depends on how many such weights the model has.
        denormals("  wq[0]", w.wq[0], dim * dim);
        denormals("  w1[0]", w.w1[0], Math.min(dim * config.hiddenDim(), 4_000_000));
        denormals("  embeddings", w.token_embedding_table, Math.min(dim * 1000, 2_000_000));
    }

    private static void denormals(String label, FloatTensor t, int count) {
        long denormal = 0;
        long nonZero = 0;
        double sumAbsDenormal = 0;
        double sumAbs = 0;
        for (int i = 0; i < count; i++) {
            float v = t.getFloat(i);
            double a = Math.abs(v);
            if (a > 0) {
                nonZero++;
                sumAbs += a;
                if (a < 6.103515625e-5) { // smallest normal FP16
                    denormal++;
                    sumAbsDenormal += a;
                }
            }
        }
        System.out.printf(
                "%-14s denormal-FP16 weights: %d/%d (%.3f%%)  their share of |w|: %.3g%%%n",
                label,
                denormal,
                count,
                100.0 * denormal / count,
                sumAbs > 0 ? 100.0 * sumAbsDenormal / sumAbs : 0.0);
    }

    private Fp16Sensitivity() {}
}
