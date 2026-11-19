package org.beehive.jitllm.backend.tornado;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

import java.nio.file.Path;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.loader.Glm5NextModelLoader;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.runtime.backend.DeviceSplit;
import org.junit.Test;

public class LayerWeightBytesAccelTest {

    private static final Pattern BLOCK = Pattern.compile("blk\\.(\\d+)\\.");

    /**
     * Every byte of a layer's device weights is found, and only once: glm5next keeps every tensor
     * in its file representation, so a layer holds on the device what its {@code blk.N} tensors
     * take in the file, with its experts loaded as several stacks.
     */
    @Test
    public void aLayerHoldsWhatItsFileBlockHolds() throws Exception {
        for (String file :
                List.of(
                        "model-Q8_0.gguf",
                        "model-medium-Q8_0.gguf",
                        "model-Q8_0-experts-Q4_0.gguf")) {
            Path path = resource(file);
            Model model = load(path);
            int layers = model.configuration().numberOfLayers();
            long[] measured = LayerWeightBytes.of((TornadoWeights) model.weights(), layers);
            assertArrayEquals(file, fileBlockBytes(path, layers), measured);
        }
    }

    /** With dense layers in front of larger expert layers, the first stage takes more layers. */
    @Test
    public void theFirstStageTakesTheSmallDenseLayers() throws Exception {
        Model model = load(resource("model-medium-Q8_0.gguf"));
        int layers = model.configuration().numberOfLayers();
        long[] bytes = LayerWeightBytes.of((TornadoWeights) model.weights(), layers);
        assertTrue(
                "dense layer 0 is not smaller than layer " + (layers - 1),
                bytes[0] < bytes[layers - 1]);
        var split = new DeviceSplit(List.of("0:0", "0:1"), List.of(), DeviceSplit.Transport.HOST);
        int[] bounds = TornadoVMMasterPlanPipeline.stageBounds(model, split, layers);
        int[] equal = split.layerBounds(layers, 1);
        assertTrue(
                "first stage " + bounds[1] + " layers, equal split " + equal[1],
                bounds[1] >= equal[1]);
    }

    private static Model load(Path path) throws Exception {
        System.setProperty(Glm5NextModelLoader.STACK_BYTES_PROPERTY, "300000");
        System.setProperty("use.tornadovm", "true");
        try {
            return ModelLoader.loadModel(path, 64, true, true);
        } finally {
            System.clearProperty(Glm5NextModelLoader.STACK_BYTES_PROPERTY);
        }
    }

    private static long[] fileBlockBytes(Path path, int layers) throws Exception {
        long[] bytes = new long[layers];
        GGUF gguf = GGUF.loadGGUFMetadata(path);
        for (GGUF.GGUFTensorInfo info : gguf.getTensorInfos().values()) {
            Matcher m = BLOCK.matcher(info.name());
            if (!m.lookingAt()) {
                continue;
            }
            int layer = Integer.parseInt(m.group(1));
            if (layer < layers) {
                long elements = 1;
                for (int d : info.dimensions()) {
                    elements *= d;
                }
                bytes[layer] +=
                        elements / info.ggmlType().getBlockSize() * info.ggmlType().getTypeSize();
            }
        }
        return bytes;
    }

    private static Path resource(String file) throws Exception {
        return Path.of(
                LayerWeightBytesAccelTest.class
                        .getResource("/goldens/glm5next-tiny/" + file)
                        .toURI());
    }
}
