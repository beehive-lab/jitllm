package org.beehive.jitllm.backend.cpu;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Map;
import org.beehive.jitllm.format.GGMLTensorEntry;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.inference.state.Glm5NextState;
import org.beehive.jitllm.inference.weights.standard.Glm5NextStandardWeights;
import org.beehive.jitllm.model.glm5next.Glm5Next;
import org.beehive.jitllm.model.glm5next.Glm5NextConfiguration;
import org.beehive.jitllm.model.loader.Glm5NextModelLoader;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.junit.Test;

/**
 * The {@code glm5next} host forward pass against goldens from the PyTorch reference.
 *
 * <p>{@code scripts/glm5next/reference.py} builds a tiny model with every layer kind of
 * GLM-5.3-Flash — KDA, sparse attention whose indexer drops positions, dense and routed
 * feed-forwards, four hyper-connection streams and the MTP layer — and records every layer's
 * streams over a 40-token sequence. {@code convert.py} wrote the model in F32 and Q8_0. The F32
 * file must match layer by layer to float rounding (the reference runs in float64); the Q8_0 file,
 * whose 64-wide rows are each a couple of blocks, only to its quantization error, so it is held to
 * the greedy choices.
 */
public class Glm5NextForwardTest {

    private static final String DIR = "/goldens/glm5next-tiny/";

    @Test
    public void f32ModelMatchesTheReferenceLayerByLayer() throws Exception {
        Run run = new Run("model-F32.gguf");
        for (int t = 0; t < run.tokens.length; t++) {
            Glm5NextForward.embed(run.config, run.weights, run.state, run.tokens[t]);
            for (int l = 0; l < run.config.numberOfLayers(); l++) {
                Glm5NextForward.layer(run.config, run.weights, run.state, l, t);
                if (!run.config.isKda(l)) {
                    assertArrayEquals(
                            "layer " + l + " position " + t + " attends to",
                            run.goldenSelection("blk." + l + ".topk", t),
                            Arrays.copyOf(run.state.selected, run.state.selectedCount));
                }
                run.compare("blk." + l + ".streams", t, run.state.streams, 1e-4f);
            }
            run.compare(
                    "logits", t, Glm5NextForward.head(run.config, run.weights, run.state), 1e-4f);
            if (t + 1 < run.tokens.length) {
                FloatTensor mtp =
                        Glm5NextForward.forwardNextn(
                                run.config, run.weights, run.state, run.tokens[t + 1], t);
                run.compare("mtp.logits", t, mtp, 1e-3f);
            }
        }
    }

    @Test
    public void q8ModelStaysCloseToTheReference() throws Exception {
        Run run = new Run("model-Q8_0.gguf");
        int agree = 0;
        for (int t = 0; t < run.tokens.length; t++) {
            FloatTensor logits =
                    Glm5NextForward.forward(run.config, run.weights, run.state, run.tokens[t], t);
            run.compare("logits", t, logits, 0.25f);
            float[] golden = run.row("logits", t);
            if (argmax(golden) == argmax(logits, golden.length)) {
                agree++;
            }
        }
        assertTrue("greedy choices agree at " + agree, agree >= run.tokens.length * 9 / 10);
    }

    @Test
    public void resetClearsTheRecurrentState() throws Exception {
        Run run = new Run("model-F32.gguf");
        float[] first = run.logitsAt(0);
        for (int t = 1; t < 5; t++) {
            Glm5NextForward.forward(run.config, run.weights, run.state, run.tokens[t], t);
        }
        run.state.resetSequenceState();
        assertArrayEquals(first, run.logitsAt(0), 0f);
    }

    private static int argmax(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }
        return best;
    }

    private static int argmax(FloatTensor values, int size) {
        int best = 0;
        for (int i = 1; i < size; i++) {
            if (values.getFloat(i) > values.getFloat(best)) {
                best = i;
            }
        }
        return best;
    }

    /** One model file, a fresh session over it, and the goldens. */
    private static final class Run {
        final Glm5NextConfiguration config;
        final Glm5NextStandardWeights weights;
        final Glm5NextState state;
        final int[] tokens;
        private final Map<String, GGMLTensorEntry> goldens;

        Run(String modelFile) throws IOException, URISyntaxException {
            Path model = resource(modelFile);
            GGUF gguf = GGUF.loadGGUFMetadata(model);
            Glm5Next glm =
                    new Glm5NextModelLoader(gguf.getFileChannel(), gguf, 64, false).loadModel();
            this.config = glm.configuration();
            this.weights = (Glm5NextStandardWeights) glm.weights();
            this.state = (Glm5NextState) glm.createNewState();

            Path golden = resource("goldens.gguf");
            GGUF goldenFile = GGUF.loadGGUFMetadata(golden);
            this.tokens = (int[]) goldenFile.getMetadata().get("golden.tokens");
            FileChannel channel = FileChannel.open(golden, StandardOpenOption.READ);
            this.goldens =
                    GGUF.loadTensorsStandard(
                            channel, goldenFile.getTensorDataOffset(), goldenFile.getTensorInfos());
        }

        float[] logitsAt(int position) {
            FloatTensor logits =
                    Glm5NextForward.forward(config, weights, state, tokens[position], position);
            float[] out = new float[config.vocabularySize()];
            for (int i = 0; i < out.length; i++) {
                out[i] = logits.getFloat(i);
            }
            return out;
        }

        /** Row {@code t} of a golden whose first (slowest) axis is the position. */
        float[] row(String name, int t) {
            GGMLTensorEntry entry = goldens.get(name);
            FloatTensor tensor = ModelLoader.loadTensor(entry);
            int[] shape = entry.shape();
            int width = 1;
            for (int d = 0; d < shape.length - 1; d++) {
                width *= shape[d];
            }
            float[] out = new float[width];
            for (int i = 0; i < width; i++) {
                out[i] = tensor.getFloat(t * width + i);
            }
            return out;
        }

        int[] goldenSelection(String name, int t) {
            return Arrays.stream(toDoubles(row(name, t)))
                    .filter(v -> v >= 0)
                    .mapToInt(v -> (int) v)
                    .toArray();
        }

        /** Max absolute difference, relative to the golden row's largest magnitude. */
        void compare(String name, int t, FloatTensor actual, float tolerance) {
            float[] golden = row(name, t);
            float scale = 0f;
            float worst = 0f;
            for (int i = 0; i < golden.length; i++) {
                scale = Math.max(scale, Math.abs(golden[i]));
                worst = Math.max(worst, Math.abs(golden[i] - actual.getFloat(i)));
            }
            float relative = worst / Math.max(scale, 1e-30f);
            assertTrue(
                    name + " at position " + t + ": relative error " + relative,
                    relative <= tolerance);
            assertEquals(name + " is finite", false, Float.isNaN(relative));
        }

        private static double[] toDoubles(float[] values) {
            double[] out = new double[values.length];
            for (int i = 0; i < values.length; i++) {
                out[i] = values[i];
            }
            return out;
        }

        private static Path resource(String file) throws URISyntaxException {
            return Path.of(Glm5NextForwardTest.class.getResource(DIR + file).toURI());
        }
    }
}
