package org.beehive.jllm.golden;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.inference.sampler.Sampler;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.standard.StandardWeights;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.format.ChatFormat;
import org.beehive.jllm.model.loader.ModelLoader;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tensor.standard.FloatTensor;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * The full-model parity test can only say the paths disagree by ~3.5 at the logits. This compares,
 * at layer 0 of the first generated position, each intermediate the GPU exposes under {@code
 * -Djllm.diag.transfers} against the same quantity recomputed on the CPU:
 *
 * <ol>
 *   <li>the RMS scale ({@code state.workspace.temp[0]}) — a pure FP32 reduction, so a difference
 *       here is accumulation order;
 *   <li>the normalized activation ({@code wrapXbFP16}) against the CPU's FP32 value <b>and</b>
 *       against that value rounded to FP16 — if the GPU matches the rounded form, the storage
 *       format is the whole difference;
 *   <li>the QKV output ({@code wrapQ}) against the CPU matmul driven by the CPU's activation and,
 *       separately, driven by the GPU's own FP16 activation. The second isolates the matmul from
 *       its input: if it matches, the projection is faithful and the error was inherited.
 * </ol>
 *
 * <p>Run with {@code -Djllm.diag.transfers=true -Djllm.diag.layer=0}.
 */
public final class LayerParity {

    public static void main(String[] args) throws Exception {
        if (!Boolean.getBoolean("jllm.diag.transfers")) {
            throw new IllegalStateException("run with -Djllm.diag.transfers=true");
        }
        Path modelPath = Path.of(System.getProperty("parity.model"));

        // ---- GPU: teacher-forced to the first generated position, layer-0 buffers read back.
        Model gpuModel = ModelLoader.loadModel(modelPath, GoldenCapture.CONTEXT_LENGTH, true, true);
        State gpuState = gpuModel.createNewState();
        List<Integer> prompt = promptTokens(gpuModel);
        TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(gpuState, gpuModel);
        int token;
        float[] gpuTemp;
        float[] gpuXb;
        float[] gpuQ;
        try {
            List<Integer> sink = new ArrayList<>();
            Sampler greedy =
                    t -> {
                        int tok = Sampler.TENSOR_ARGMAX.sampleToken(t);
                        sink.add(tok);
                        return tok;
                    };
            gpuModel.generateTokensGPU(
                    gpuState, 0, prompt, Set.of(), prompt.size() + 1, greedy, false, null, plan);
            token = sink.get(0);
            org.beehive.jllm.backend.tornado.TornadoForwardPass.forward(
                    gpuModel, gpuState, token, prompt.size(), plan);
            gpuTemp = snap(gpuState.workspace.temp);
            gpuXb = snapHalf(gpuState.workspace.wrapXbFP16);
            gpuQ = snap(gpuState.workspace.wrapQ);
        } finally {
            plan.freeTornadoExecutionPlan();
        }

        // ---- CPU: the same layer-0 quantities, recomputed from the standard weights.
        Model cpuModel =
                ModelLoader.loadModel(modelPath, GoldenCapture.CONTEXT_LENGTH, true, false);
        StandardWeights w = (StandardWeights) cpuModel.weights();
        int dim = cpuModel.configuration().dim();
        float eps = cpuModel.configuration().rmsNormEps();

        float[] x = new float[dim];
        FloatTensor xT = new ArrayFloatTensor(x);
        w.token_embedding_table.copyTo(token * dim, xT, 0, dim);

        double ss = 0;
        for (float v : x) {
            ss += (double) v * v;
        }
        float cpuScale = (float) (1.0 / Math.sqrt(ss / dim + eps));

        float[] cpuXb = new float[dim];
        for (int i = 0; i < dim; i++) {
            cpuXb[i] = w.rms_att_weight[0].getFloat(i) * (cpuScale * x[i]);
        }
        float[] cpuXbAsFp16 = new float[dim];
        for (int i = 0; i < dim; i++) {
            cpuXbAsFp16[i] = new HalfFloat(cpuXb[i]).getFloat32();
        }

        int qDim = gpuQ.length;
        float[] cpuQ = new float[qDim];
        w.wq[0].matmul(new ArrayFloatTensor(cpuXb), new ArrayFloatTensor(cpuQ), qDim, dim);
        float[] cpuQFromGpuXb = new float[qDim];
        w.wq[0].matmul(new ArrayFloatTensor(gpuXb), new ArrayFloatTensor(cpuQFromGpuXb), qDim, dim);

        // wrapQ is read back at the end of the layer graph, i.e. after rope_and_kv_cache has
        // rotated it in place, so the CPU side has to be rotated too before the two are comparable.
        int headSize = cpuModel.configuration().headSize();
        int position = prompt.size();
        float[] cpuQRope = rope(cpuQ, w, position, headSize);
        float[] cpuQFromGpuXbRope = rope(cpuQFromGpuXb, w, position, headSize);

        System.out.printf(
                "token=%d position=%d dim=%d qDim=%d headSize=%d ropeTheta=%s%n",
                token,
                position,
                dim,
                qDim,
                headSize,
                cpuModel.configuration()
                                instanceof org.beehive.jllm.model.llama.LlamaConfiguration lc
                        ? Float.toString(lc.ropeTheta())
                        : "n/a");
        System.out.printf(
                "rms scale: cpu=%.9g gpu=%.9g absDiff=%.3g relDiff=%.3g%n",
                cpuScale,
                gpuTemp[0],
                Math.abs(cpuScale - gpuTemp[0]),
                Math.abs(cpuScale - gpuTemp[0]) / Math.abs(cpuScale));
        report("xb: GPU FP16 vs CPU FP32", cpuXb, gpuXb);
        report("xb: GPU FP16 vs CPU rounded to FP16", cpuXbAsFp16, gpuXb);
        report("Q pre-RoPE : GPU vs CPU(CPU xb)", cpuQ, gpuQ);
        report("Q post-RoPE: GPU vs CPU(CPU xb)", cpuQRope, gpuQ);
        report("Q post-RoPE: GPU vs CPU(GPU xb)", cpuQFromGpuXbRope, gpuQ);
        // Rotation-invariant: if the pair magnitudes agree but the values do not, the projection is
        // fine and the two paths disagree about the rotation (angle, pairing, or position).
        report("Q pair magnitudes (RoPE-invariant)", pairNorms(cpuQ), pairNorms(gpuQ));
    }

    private static float[] rope(float[] q, StandardWeights w, int position, int headSize) {
        float[] out = q.clone();
        for (int i = 0; i < out.length; i += 2) {
            int headDim = i % headSize;
            float fcr = w.freq_cis_real.getFloat(position * (headSize / 2) + (headDim / 2));
            float fci = w.freq_cis_imag.getFloat(position * (headSize / 2) + (headDim / 2));
            float v0 = out[i];
            float v1 = out[i + 1];
            out[i] = v0 * fcr - v1 * fci;
            out[i + 1] = v0 * fci + v1 * fcr;
        }
        return out;
    }

    private static float[] pairNorms(float[] v) {
        float[] out = new float[v.length / 2];
        for (int i = 0; i < out.length; i++) {
            out[i] = (float) Math.hypot(v[2 * i], v[2 * i + 1]);
        }
        return out;
    }

    private static void report(String label, float[] ref, float[] got) {
        double maxAbs = 0, sumSqDiff = 0, sumSqRef = 0, sumAbs = 0;
        int exactBits = 0;
        double maxRel = 0;
        for (int i = 0; i < ref.length; i++) {
            double d = Math.abs((double) ref[i] - got[i]);
            maxAbs = Math.max(maxAbs, d);
            sumAbs += d;
            sumSqDiff += d * d;
            sumSqRef += (double) ref[i] * ref[i];
            if (Math.abs(ref[i]) > 1e-6) {
                maxRel = Math.max(maxRel, d / Math.abs(ref[i]));
            }
            if (Float.floatToRawIntBits(ref[i]) == Float.floatToRawIntBits(got[i])) {
                exactBits++;
            }
        }
        System.out.printf(
                "%-38s maxAbs=%.6g meanAbs=%.6g relL2=%.6g maxRel=%.4g bitIdentical=%d/%d%n",
                label,
                maxAbs,
                sumAbs / ref.length,
                Math.sqrt(sumSqDiff / sumSqRef),
                maxRel,
                exactBits,
                ref.length);
    }

    private static List<Integer> promptTokens(Model model) {
        ChatFormat cf = model.chatFormat();
        List<Integer> prompt = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            prompt.add(cf.getBeginOfText());
        }
        prompt.addAll(
                cf.encodeMessage(
                        new ChatFormat.Message(ChatFormat.Role.USER, GoldenCapture.PROMPT)));
        prompt.addAll(cf.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        return prompt;
    }

    private static float[] snap(FloatArray a) {
        float[] out = new float[a.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = a.get(i);
        }
        return out;
    }

    private static float[] snapHalf(HalfFloatArray a) {
        float[] out = new float[a.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = a.get(i).getFloat32();
        }
        return out;
    }

    private LayerParity() {}
}
