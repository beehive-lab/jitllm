package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.EnumMap;
import java.util.Map;
import org.beehive.jitllm.format.DataTypeMapping;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.runtime.memory.WeightFootprint;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * The device weight prediction for a genuinely heterogeneous model, per representation.
 *
 * <p>Qwen3.8-27B is mixed by construction — Q4_0 projections and token embeddings, eight Q4_1
 * {@code ffn_down}, forty-eight Q5_K {@code ssm_out}, a Q6_K vocabulary projection, a Q8_0 MTP
 * projection, and F32 norms, SSM parameters and convolution kernels. A prediction derived from one
 * model-wide dtype is wrong for every tensor that is not that dtype, and the error runs to
 * gigabytes.
 *
 * <p>The expectations here are computed from block arithmetic per representation rather than copied
 * from a run, so this validates each dtype's accounting rather than being tuned to one file.
 */
public class Qwen35WeightFootprintTest {

    /** Every representation the device holds, as (weights per block, bytes per block). */
    private static final Map<DataType, int[]> BLOCK = new EnumMap<>(DataType.class);

    static {
        BLOCK.put(DataType.F32, new int[] {1, 4});
        BLOCK.put(DataType.F16, new int[] {1, 2});
        BLOCK.put(DataType.Q4_0, new int[] {32, 18});
        BLOCK.put(DataType.Q4_1, new int[] {32, 20});
        BLOCK.put(DataType.Q8_0, new int[] {32, 34});
        BLOCK.put(DataType.Q4_K, new int[] {256, 144});
        BLOCK.put(DataType.Q5_K, new int[] {256, 176});
        BLOCK.put(DataType.Q6_K, new int[] {256, 210});
    }

    private static Path fixture() {
        return GoldenFixture.locate(GoldenFixture.Fixture.QWEN3_8_27B_Q4_0);
    }

    /**
     * The prediction is the file's own weight bytes.
     *
     * <p>14.944 GiB, every quantized tensor at its own block size; converted to Q8_0, as the device
     * used to hold it, the same model was about 27 GiB, which does not fit the 24 GiB device it
     * runs on. Asserted as a range so a future re-quantization of the fixture fails loudly rather
     * than drifting.
     */
    @Test
    public void aMixedModelIsPredictedAtEachTensorsOwnRepresentation() throws Exception {
        Path model = fixture();
        assumeTrue(
                GoldenFixture.absentMessage(GoldenFixture.Fixture.QWEN3_8_27B_Q4_0), model != null);

        WeightFootprint predicted = ModelLoader.weightFootprint(model);
        long total = predicted.perLayerBytes() + predicted.globalBytes();
        assertTrue(
                "prediction was " + total + " bytes",
                total > 15_900_000_000L && total < 16_100_000_000L);
    }

    /**
     * Each representation's accounting on its own: the prediction equals, per layer and globally,
     * the sum over the file's descriptors of each tensor's own block arithmetic. A predictor that
     * got one format's block size wrong would pass the whole-model range and fail here.
     */
    @Test
    public void everyTensorIsCountedAtItsOwnBlockSize() throws Exception {
        Path model = fixture();
        assumeTrue(
                GoldenFixture.absentMessage(GoldenFixture.Fixture.QWEN3_8_27B_Q4_0), model != null);

        long perLayer = 0;
        long global = 0;
        for (GGUF.GGUFTensorInfo info : GGUF.loadGGUFMetadata(model).getTensorInfos().values()) {
            if (info.name().equals("rope_freqs.weight")) {
                continue;
            }
            long elements = 1;
            for (int d : info.dimensions()) {
                elements *= d;
            }
            int[] block = BLOCK.get(DataTypeMapping.sourceType(info.ggmlType()).narrowedFallback());
            long bytes = elements / block[0] * block[1];
            if (info.name().startsWith("blk.")) {
                perLayer += bytes;
            } else {
                global += bytes;
            }
        }

        WeightFootprint predicted = ModelLoader.weightFootprint(model);
        assertEquals("per layer", perLayer, predicted.perLayerBytes());
        assertEquals("global", global, predicted.globalBytes());
    }
}
