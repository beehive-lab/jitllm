package org.beehive.jitllm.backend.cpu;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import org.beehive.jitllm.format.GGMLTensorEntry;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.golden.Fp32KeyValueCache;
import org.beehive.jitllm.inference.state.DeepSeek4State;
import org.beehive.jitllm.inference.weights.standard.DeepSeek4StandardWeights;
import org.beehive.jitllm.model.deepseek4.DeepSeek4;
import org.beehive.jitllm.model.deepseek4.DeepSeek4Configuration;
import org.beehive.jitllm.model.loader.DeepSeek4ModelLoader;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.junit.ClassRule;
import org.junit.Test;

/**
 * The {@code deepseek4} host forward pass against goldens from DeepSeek's own inference code.
 *
 * <p>{@code scripts/deepseek4/reference.py} builds a tiny model with every layer kind of
 * DeepSeek-V4 — hash and score routing, compression ratios 0, 4 (with the indexer, which keeps 4 of
 * up to 75 blocks) and 128, four hyper-connection streams — runs the official {@code model.py} over
 * 300 tokens, and records every layer's streams and the logits. {@code convert.py} wrote the model
 * in F32 and Q8_0. The F32 file must match layer by layer to float rounding; the Q8_0 file only to
 * its quantization error, so it is held to the greedy choices. The reference keeps its window in
 * FP32, so the window cache is FP32 here too.
 */
public class DeepSeek4ForwardTest {

    @ClassRule public static final Fp32KeyValueCache FP32 = new Fp32KeyValueCache();

    private static final String DIR = "/goldens/deepseek4-tiny/";

    /** Loaded as two stacks of four experts, as the full model's experts load as two stacks. */
    @Test
    public void f32ModelMatchesTheReferenceLayerByLayer() throws Exception {
        Run run = new Run("model-F32.gguf", 70_000);
        assertEquals(4, run.config.expertChunkSize());
        for (int t = 0; t < run.tokens.length; t++) {
            DeepSeek4Forward.embed(run.config, run.weights, run.state, run.tokens[t]);
            for (int l = 0; l < run.config.numberOfLayers(); l++) {
                DeepSeek4Forward.layer(run.config, run.weights, run.state, l, run.tokens[t], t);
                run.compare("layer" + l + ".streams", t, run.state.streams, 2e-4f);
            }
            run.compare(
                    "logits", t, DeepSeek4Forward.head(run.config, run.weights, run.state), 2e-4f);
        }
    }

    @Test
    public void q8ModelStaysCloseToTheReference() throws Exception {
        Run run = new Run("model-Q8_0.gguf");
        int agree = 0;
        for (int t = 0; t < run.tokens.length; t++) {
            FloatTensor logits =
                    DeepSeek4Forward.forward(run.config, run.weights, run.state, run.tokens[t], t);
            float[] golden = run.row("logits", t);
            if (argmax(golden) == argmax(logits, golden.length)) {
                agree++;
            }
        }
        assertTrue("greedy choices agree at " + agree, agree >= run.tokens.length * 8 / 10);
    }

    @Test
    public void resetClearsThePendingCompressorEntries() throws Exception {
        Run run = new Run("model-F32.gguf");
        float[] first = new float[run.config.vocabularySize()];
        for (int t = 0; t < 7; t++) {
            float[] logits = run.logitsAt(t);
            if (t == 0) {
                first = logits;
            }
        }
        run.state.resetSequenceState();
        float[] again = run.logitsAt(0);
        run.state.resetSequenceState();
        float[] pending = run.logitsAt(0);
        for (int t = 1; t < 4; t++) {
            run.logitsAt(t);
        }
        assertArrayEquals(first, again, 0f);
        assertArrayEquals(first, pending, 0f);
        run.compare("logits", 3, run.state.logits, 2e-4f);
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
        final DeepSeek4Configuration config;
        final DeepSeek4StandardWeights weights;
        final DeepSeek4State state;
        final int[] tokens;
        private final Map<String, GGMLTensorEntry> goldens;

        Run(String modelFile) throws IOException, URISyntaxException {
            this(modelFile, 0);
        }

        /** With {@code stackBytes > 0}, the experts loaded as stacks of at most that size. */
        Run(String modelFile, long stackBytes) throws IOException, URISyntaxException {
            Path model = resource(modelFile);
            GGUF gguf = GGUF.loadGGUFMetadata(model);
            if (stackBytes > 0) {
                System.setProperty(
                        DeepSeek4ModelLoader.STACK_BYTES_PROPERTY, Long.toString(stackBytes));
            }
            DeepSeek4 deepseek;
            try {
                deepseek =
                        new DeepSeek4ModelLoader(gguf.getFileChannel(), gguf, 512, false)
                                .loadModel();
            } finally {
                System.clearProperty(DeepSeek4ModelLoader.STACK_BYTES_PROPERTY);
            }
            this.config = deepseek.configuration();
            this.weights = (DeepSeek4StandardWeights) deepseek.weights();
            this.state = (DeepSeek4State) deepseek.createNewState();

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
                    DeepSeek4Forward.forward(config, weights, state, tokens[position], position);
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
            assertEquals(name + " is finite", false, Float.isNaN(relative));
            assertTrue(
                    name + " at position " + t + ": relative error " + relative,
                    relative <= tolerance);
        }

        private static Path resource(String file) throws URISyntaxException {
            return Path.of(DeepSeek4ForwardTest.class.getResource(DIR + file).toURI());
        }
    }
}
