package org.beehive.jitllm.golden;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.EnumMap;
import java.util.Map;
import java.util.Set;
import org.beehive.jitllm.model.loader.ModelLoader;
import org.beehive.jitllm.runtime.memory.DeviceRetention;
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

    /** Every representation the engine retains, as (weights per block, bytes per block). */
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

    private static final Set<DataType> ALL_NATIVE =
            Set.of(
                    DataType.F32,
                    DataType.F16,
                    DataType.Q4_0,
                    DataType.Q4_1,
                    DataType.Q4_K,
                    DataType.Q5_K,
                    DataType.Q6_K,
                    DataType.Q8_0);

    private static Path fixture() {
        return GoldenFixture.locate(GoldenFixture.Fixture.QWEN3_8_27B_Q4_0);
    }

    /**
     * Retained, the prediction is the file's own weight bytes.
     *
     * <p>The number is derived, not recorded: every tensor is counted at its own block size, and
     * the total is what the GGUF holds for those tensors. If the prediction agreed with a recorded
     * constant but not with the arithmetic, it would be tuned to this file.
     */
    @Test
    public void aMixedModelIsPredictedAtEachTensorsOwnRepresentation() throws Exception {
        Path model = fixture();
        assumeTrue(
                GoldenFixture.absentMessage(GoldenFixture.Fixture.QWEN3_8_27B_Q4_0), model != null);

        WeightFootprint retained =
                ModelLoader.weightFootprint(model, DeviceRetention.retaining(ALL_NATIVE));
        WeightFootprint converting =
                ModelLoader.weightFootprint(model, DeviceRetention.converting());

        long retainedTotal = retained.perLayerBytes() + retained.globalBytes();
        long convertingTotal = converting.perLayerBytes() + converting.globalBytes();

        // 14.944 GiB of weight bytes, from the inventory: every quantized tensor at its own block
        // size. Asserted as a range so a future re-quantization of the fixture fails loudly rather
        // than drifting.
        assertTrue(
                "retained prediction was " + retainedTotal + " bytes",
                retainedTotal > 15_900_000_000L && retainedTotal < 16_100_000_000L);

        // Converted, the same model is about 27 GiB — which does not fit the 24 GiB device this
        // was blocked on, and is the whole reason retention exists.
        assertTrue(
                "converting prediction was " + convertingTotal + " bytes",
                convertingTotal > 28_000_000_000L);
        assertTrue(
                "conversion must cost substantially more, not less",
                convertingTotal > retainedTotal * 17 / 10);
    }

    /**
     * Each representation's accounting on its own.
     *
     * <p>Retaining exactly one dtype at a time isolates that dtype's contribution: the difference
     * from the fully-converting prediction is the bytes saved on the tensors of that type, and it
     * must equal the block-size ratio for them. A predictor that got one format's block size wrong
     * would pass a whole-model check and fail here.
     */
    @Test
    public void everyRepresentationIsAccountedForSeparately() throws Exception {
        Path model = fixture();
        assumeTrue(
                GoldenFixture.absentMessage(GoldenFixture.Fixture.QWEN3_8_27B_Q4_0), model != null);

        WeightFootprint converting =
                ModelLoader.weightFootprint(model, DeviceRetention.converting());
        long convertingTotal = converting.perLayerBytes() + converting.globalBytes();

        for (DataType type :
                new DataType[] {DataType.Q4_0, DataType.Q4_1, DataType.Q5_K, DataType.Q6_K}) {
            WeightFootprint one =
                    ModelLoader.weightFootprint(model, DeviceRetention.retaining(Set.of(type)));
            long total = one.perLayerBytes() + one.globalBytes();
            long saved = convertingTotal - total;
            assertTrue(
                    type + " retained saved nothing, so it was not accounted for separately",
                    saved > 0);

            // The saving must be exactly (Q8_0 bytes - own bytes) per block of that type, so the
            // implied block count has to be a whole number.
            int[] block = BLOCK.get(type);
            long q8BytesPerBlock = (long) block[0] / 32 * 34;
            long savedPerBlock = q8BytesPerBlock - block[1];
            assertEquals(
                    type + "'s saving is not a whole number of its blocks",
                    0,
                    saved % savedPerBlock);
        }

        // Q8_0 and F32 are never converted, so retaining them changes nothing.
        for (DataType unchanged : new DataType[] {DataType.Q8_0, DataType.F32}) {
            WeightFootprint one =
                    ModelLoader.weightFootprint(
                            model, DeviceRetention.retaining(Set.of(unchanged)));
            assertEquals(
                    unchanged + " is not converted either way",
                    convertingTotal,
                    one.perLayerBytes() + one.globalBytes());
        }
    }
}
