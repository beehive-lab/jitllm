package org.beehive.jllm.golden;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.inference.sampler.Sampler;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.format.ChatFormat;
import org.beehive.jllm.model.loader.ModelLoader;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Measures the divergence <b>rate</b> of repeated identical execution on one fixed plan.
 *
 * <p>Yes/no sampling has repeatedly misled this investigation: five clean repeats were read as
 * "deterministic" three separate times and were wrong each time. This reports a rate over a stated
 * sample size so that a candidate fix can be compared against a baseline instead of against "it
 * looked fine".
 */
public final class DivergenceRate {

    public static void main(String[] args) throws Exception {
        Path modelPath = Paths.get(System.getProperty("rate.model"));
        int iterations = Integer.getInteger("rate.iterations", 200);

        Model model = ModelLoader.loadModel(modelPath, 512, true, true);
        State state = model.createNewState();

        ChatFormat cf = model.chatFormat();
        List<Integer> prompt = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            prompt.add(cf.getBeginOfText());
        }
        prompt.addAll(
                cf.encodeMessage(
                        new ChatFormat.Message(ChatFormat.Role.USER, GoldenCapture.PROMPT)));
        prompt.addAll(cf.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
        try {
            // Ingest the prompt once so the KV cache is populated and positions are realistic.
            List<Integer> sink = new ArrayList<>();
            Sampler greedy =
                    t -> {
                        int tok = Sampler.TENSOR_ARGMAX.sampleToken(t);
                        sink.add(tok);
                        return tok;
                    };
            model.generateTokensGPU(
                    state, 0, prompt, Set.of(), prompt.size() + 1, greedy, false, null, plan);

            int pos = prompt.size();
            int tok = sink.get(0);

            // Warm-up forward, discarded. Under -Djllm.diag.transfers the host copies are
            // taken at the end of layer N's graph, so the parts of wrapKeyCache belonging to layers
            // beyond N are one forward behind. Capturing the reference straight after the first
            // forward would freeze that lag into the reference and report every later iteration as
            // differing (this is the "kvCache=300/300" that looked like a defect and is not one).
            org.beehive.jllm.backend.tornado.TornadoForwardPass.forward(
                    model, state, tok, pos, plan);

            float[] ref =
                    snap(
                            org.beehive.jllm.backend.tornado.TornadoForwardPass.forward(
                                    model, state, tok, pos, plan));
            float[] refKv = snap(state.workspace.wrapKeyCache);
            float[] refX = snap(state.workspace.wrapX);
            float[] refQ = snap(state.workspace.wrapQ);
            int kvDiv = 0;
            int xDiv = 0;
            int qDiv = 0;
            float[] refXb = snapHalf(state.workspace.wrapXbFP16);
            float[] refTemp = snap(state.workspace.temp);
            int xbDiv = 0;
            int tempDiv = 0;
            int diverged = 0;
            double worst = 0;
            int firstAt = -1;
            for (int i = 1; i <= iterations; i++) {
                float[] got =
                        snap(
                                org.beehive.jllm.backend.tornado.TornadoForwardPass.forward(
                                        model, state, tok, pos, plan));
                boolean differs = false;
                for (int j = 0; j < ref.length; j++) {
                    if (Float.floatToRawIntBits(ref[j]) != Float.floatToRawIntBits(got[j])) {
                        differs = true;
                        worst = Math.max(worst, Math.abs(ref[j] - got[j]));
                    }
                }
                if (differs) {
                    diverged++;
                    if (firstAt < 0) {
                        firstAt = i;
                    }
                }
                if (!same(refKv, snap(state.workspace.wrapKeyCache))) {
                    kvDiv++;
                }
                if (!same(refX, snap(state.workspace.wrapX))) {
                    xDiv++;
                }
                if (!same(refQ, snap(state.workspace.wrapQ))) {
                    qDiv++;
                }
                if (!same(refXb, snapHalf(state.workspace.wrapXbFP16))) {
                    xbDiv++;
                }
                if (!same(refTemp, snap(state.workspace.temp))) {
                    tempDiv++;
                }
            }
            System.out.printf(
                    "  STAGE temp(rms)=%d/%d  wrapXbFP16(rms out)=%d/%d  wrapQ(qkv out)=%d/%d  kvCache=%d/%d  wrapX=%d/%d%n",
                    tempDiv,
                    iterations,
                    xbDiv,
                    iterations,
                    qDiv,
                    iterations,
                    kvDiv,
                    iterations,
                    xDiv,
                    iterations);
            System.out.printf(
                    "RATE model=%s iterations=%d diverged=%d (%.1f%%) firstAt=%d worstAbs=%.7f%n",
                    modelPath.getFileName(),
                    iterations,
                    diverged,
                    100.0 * diverged / iterations,
                    firstAt,
                    worst);
        } finally {
            plan.freeTornadoExecutionPlan();
        }
    }

    private static float[] snapHalf(uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray a) {
        float[] out = new float[a.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = a.get(i).getFloat32();
        }
        return out;
    }

    private static boolean same(float[] a, float[] b) {
        for (int i = 0; i < a.length; i++) {
            if (Float.floatToRawIntBits(a[i]) != Float.floatToRawIntBits(b[i])) {
                return false;
            }
        }
        return true;
    }

    private static float[] snap(org.beehive.jllm.inference.Logits logits) {
        float[] out = new float[logits.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = logits.get(i);
        }
        return out;
    }

    private static float[] snap(FloatArray fa) {
        float[] out = new float[fa.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = fa.get(i);
        }
        return out;
    }

    private DivergenceRate() {}
}
