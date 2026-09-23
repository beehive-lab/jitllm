package org.beehive.jitllm.golden;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.cpu.InferenceCore;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.standard.StandardWeights;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.granite.GraniteConfiguration;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import uk.ac.manchester.tornado.api.types.HalfFloat;

/**
 * Diagnostic: does the FP16 GPU path's activation format explain its distance from the CPU?
 *
 * <p>The FP16 path stores the attention block's normalized activation as FP16 before the QKV
 * projection, and the same for the final normalized activation before the vocabulary projection.
 * Everything else stays FP32. This runs the CPU forward with rounding at exactly those two points
 * and compares the result against the GPU.
 *
 * <p>If the simulated CPU lands on the GPU (~1e-5, the level the Q8_0 path already achieves), then
 * the FP16 configuration's larger CPU↔GPU gap is the format, not a defect — and a model that sits
 * further out than its peers is simply more sensitive to the same perturbation. If the simulation
 * does <em>not</em> land on the GPU, something else differs and is worth chasing.
 */
public final class Fp16SimParity {

    public static void main(String[] args) throws Exception {
        for (String spec : System.getProperty("sim.models").split(",")) {
            Path path = Path.of(spec);
            System.out.println("=== " + path.getFileName());
            GoldenCapture.Result reference = GoldenCapture.capture(path, false);
            List<float[]> gpu = GoldenCapture.capture(path, true, reference.tokenIds).rows;
            List<float[]> plain = cpuRows(path, false, reference.tokenIds);
            List<float[]> simulated = cpuRows(path, true, reference.tokenIds);
            compare("CPU (plain)     vs GPU", plain, gpu);
            compare("CPU (fp16 sim)  vs GPU", simulated, gpu);
            compare("CPU (plain)     vs CPU (fp16 sim)", plain, simulated);
        }
    }

    /**
     * Teacher-forced along {@code forced} — the same token history GoldenCapture drives the GPU
     * with — so row r on both sides is the same computation.
     */
    private static List<float[]> cpuRows(Path path, boolean simulateFp16, List<Integer> forced)
            throws Exception {
        Model model = ModelLoader.loadModel(path, GoldenCapture.CONTEXT_LENGTH, true, false);
        State state = model.createNewState();
        ChatFormat cf = model.chatFormat();

        List<Integer> tokens = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            tokens.add(cf.getBeginOfText());
        }
        tokens.addAll(
                cf.encodeMessage(
                        new ChatFormat.Message(ChatFormat.Role.USER, GoldenCapture.PROMPT)));
        tokens.addAll(cf.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        // Mirrors TokenGenerationLoop.generateTokens*: the first forward is of state.latestToken,
        // each prompt token is fed on the following step, and the first captured row is the forward
        // that happens once the prompt is exhausted. Getting this shifted by one compares different
        // computations and shows up as a ~0.5 relative L2 that looks like a defect.
        List<float[]> rows = new ArrayList<>();
        int currentToken = state.latestToken;
        int promptIndex = 0;
        int pos = 0;
        while (rows.size() < GoldenCapture.TOKENS) {
            float[] logits = forward(model, state, currentToken, pos++, simulateFp16);
            if (promptIndex < tokens.size()) {
                currentToken = tokens.get(promptIndex++);
            } else {
                rows.add(logits);
                currentToken = forced.get(rows.size() - 1);
            }
        }
        return rows;
    }

    private static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }

    private static int argmax(FloatTensor t) {
        int best = 0;
        for (int i = 1; i < t.size(); i++) {
            if (t.getFloat(i) > t.getFloat(best)) {
                best = i;
            }
        }
        return best;
    }

    /**
     * One decode step. Mirrors {@link InferenceCore#forwardGranite} (which is the Llama forward
     * plus the µP multipliers; for a non-Granite model the multipliers are the identity), with
     * optional FP16 rounding where the GPU stores FP16.
     */
    private static float[] forward(
            Model model, State state, int token, int position, boolean simulateFp16) {
        Configuration config = model.configuration();
        StandardWeights w = (StandardWeights) model.weights();
        int dim = config.dim();
        int headSize = config.headSize();
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int kvMul = config.numberOfHeads() / config.numberOfKeyValueHeads();

        boolean granite = config instanceof GraniteConfiguration;
        float embeddingScale = granite ? ((GraniteConfiguration) config).embeddingScale() : 1.0f;
        float residualScale = granite ? ((GraniteConfiguration) config).residualScale() : 1.0f;
        float attentionScale =
                granite
                        ? ((GraniteConfiguration) config).attentionScale()
                        : (float) (1.0 / Math.sqrt(headSize));
        float logitScale = granite ? ((GraniteConfiguration) config).logitScale() : 1.0f;

        w.token_embedding_table.copyTo(token * dim, state.x, 0, dim);
        if (granite) {
            state.x.mapInPlace(v -> v * embeddingScale);
        }

        for (int l = 0; l < config.numberOfLayers(); l++) {
            InferenceCore.rmsnorm(
                    state.xb, state.x, w.rms_att_weight[l], 0, dim, config.rmsNormEps());
            if (simulateFp16) {
                roundToFp16(state.xb, dim); // GPU: attn_rms_apply_fp16 → wrapXbFP16
            }

            w.wq[l].matmul(state.xb, state.q, dim, dim);
            w.wk[l].matmul(state.xb, state.k, kvDim, dim);
            w.wv[l].matmul(state.xb, state.v, kvDim, dim);

            for (int i = 0; i < dim; i += 2) {
                int headDim = i % headSize;
                float fcr = w.freq_cis_real.getFloat(position * (headSize / 2) + (headDim / 2));
                float fci = w.freq_cis_imag.getFloat(position * (headSize / 2) + (headDim / 2));
                int rotn = i < kvDim ? 2 : 1;
                for (int v = 0; v < rotn; v++) {
                    FloatTensor vec = v == 0 ? state.q : state.k;
                    float v0 = vec.getFloat(i);
                    float v1 = vec.getFloat(i + 1);
                    vec.setFloat(i, v0 * fcr - v1 * fci);
                    vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                }
            }

            state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

            for (int h = 0; h < config.numberOfHeads(); h++) {
                int qOffset = h * headSize;
                int attOffset = h * config.contextLength();
                for (int t = 0; t <= position; t++) {
                    int keyOffset = t * kvDim + (h / kvMul) * headSize;
                    float score =
                            state.q.dot(qOffset, state.keyCache[l], keyOffset, headSize)
                                    * attentionScale;
                    state.att.setFloat(attOffset + t, score);
                }
                state.att.softmaxInPlace(attOffset, position + 1);
                int xbOffset = h * headSize;
                state.xb.fillInPlace(xbOffset, headSize, 0f);
                for (int t = 0; t <= position; t++) {
                    int vOffset = t * kvDim + (h / kvMul) * headSize;
                    float a = state.att.getFloat(attOffset + t);
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[l], vOffset, headSize, a);
                }
            }

            w.wo[l].matmul(state.xb, state.xb2, dim, dim);
            if (granite) {
                state.xb2.mapInPlace(v -> v * residualScale);
            }
            state.x.addInPlace(state.xb2);

            InferenceCore.rmsnorm(
                    state.xb, state.x, w.rms_ffn_weight[l], 0, dim, config.rmsNormEps());
            // The GPU fuses the FFN normalization into the gate/up projection and reads FP32, so no
            // rounding here — that asymmetry is itself part of what is being checked.

            w.w1[l].matmul(state.xb, state.hb, config.hiddenDim(), dim);
            w.w3[l].matmul(state.xb, state.hb2, config.hiddenDim(), dim);
            state.hb.mapInPlace(v -> v / (float) (1.0 + Math.exp(-v)));
            state.hb.multiplyInPlace(state.hb2);
            w.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim());
            if (granite) {
                state.xb.mapInPlace(v -> v * residualScale);
            }
            state.x.addInPlace(state.xb);
        }

        InferenceCore.rmsnorm(state.x, state.x, w.rms_final_weight, 0, dim, config.rmsNormEps());
        if (simulateFp16) {
            roundToFp16(state.x, dim); // GPU: logits.rms_apply_fp16 → wrapXbFP16
        }
        w.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);
        if (granite) {
            state.logits.mapInPlace(v -> v * logitScale);
        }

        float[] out = new float[config.vocabularySize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = state.logits.getFloat(i);
        }
        return out;
    }

    private static void roundToFp16(FloatTensor t, int size) {
        for (int i = 0; i < size; i++) {
            t.setFloat(i, new HalfFloat(t.getFloat(i)).getFloat32());
        }
    }

    private static void compare(String label, List<float[]> a, List<float[]> b) {
        int n = Math.min(a.size(), b.size());
        double worstRelL2 = 0;
        double maxAbs = 0;
        int disagreements = 0;
        for (int r = 0; r < n; r++) {
            float[] x = a.get(r);
            float[] y = b.get(r);
            double sqDiff = 0;
            double sqRef = 0;
            for (int i = 0; i < x.length; i++) {
                double d = (double) x[i] - y[i];
                sqDiff += d * d;
                sqRef += (double) x[i] * x[i];
                maxAbs = Math.max(maxAbs, Math.abs(d));
            }
            worstRelL2 = Math.max(worstRelL2, Math.sqrt(sqDiff / sqRef));
            if (argmax(x) != argmax(y)) {
                disagreements++;
            }
        }
        System.out.printf(
                "  %-34s worst relL2=%.4g  maxAbs=%.4g  argmax diffs=%d/%d%n",
                label, worstRelL2, maxAbs, disagreements, n);
    }

    private Fp16SimParity() {}
}
