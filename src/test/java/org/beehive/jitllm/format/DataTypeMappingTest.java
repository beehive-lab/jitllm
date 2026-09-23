package org.beehive.jitllm.format;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.ExecutionTarget;
import org.junit.Test;

/**
 * This logic existed before as one untested switch with a GPU answer and no CPU half. The point of
 * these tests is not that a mapping exists but that the two halves differ: a K-quant is a K-quant
 * on the host and becomes Q8_0 on the device, and a test that only checked one target would pass
 * while the other was silently wrong.
 */
public class DataTypeMappingTest {

    private static final GGMLType[] RECOGNIZED = {
        GGMLType.F32,
        GGMLType.F16,
        GGMLType.Q8_0,
        GGMLType.Q4_0,
        GGMLType.Q4_K,
        GGMLType.Q5_K,
        GGMLType.Q6_K
    };

    private static final GGMLType[] DEVICE_MATERIALIZED = {
        GGMLType.Q4_0, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K
    };

    @Test
    public void theSourceTypeIsWhatIsInTheFile() {
        assertSame(DataType.F32, DataTypeMapping.sourceType(GGMLType.F32));
        assertSame(DataType.F16, DataTypeMapping.sourceType(GGMLType.F16));
        assertSame(DataType.Q8_0, DataTypeMapping.sourceType(GGMLType.Q8_0));
        assertSame(DataType.Q4_0, DataTypeMapping.sourceType(GGMLType.Q4_0));
        assertSame(DataType.Q4_K, DataTypeMapping.sourceType(GGMLType.Q4_K));
        assertSame(DataType.Q5_K, DataTypeMapping.sourceType(GGMLType.Q5_K));
        assertSame(DataType.Q6_K, DataTypeMapping.sourceType(GGMLType.Q6_K));
    }

    @Test
    public void anUnexecutableFormatIsRefusedByName() {
        try {
            DataTypeMapping.sourceType(GGMLType.Q2_K);
            fail("a format nothing executes must not be silently mapped to something else");
        } catch (UnsupportedOperationException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("Q2_K"));
        }
    }

    @Test
    public void theHostExecutesEveryRecognizedRepresentationAsItIs() {
        for (GGMLType fileType : RECOGNIZED) {
            assertSame(
                    "the CPU decodes " + fileType + " during compute, so nothing is converted",
                    DataTypeMapping.sourceType(fileType),
                    DataTypeMapping.materializedType(fileType, ExecutionTarget.CPU));
        }
    }

    @Test
    public void theDeviceKeepsWhatItHasKernelsFor() {
        assertSame(
                DataType.F32, DataTypeMapping.materializedType(GGMLType.F32, ExecutionTarget.GPU));
        assertSame(
                DataType.F16, DataTypeMapping.materializedType(GGMLType.F16, ExecutionTarget.GPU));
        assertSame(
                DataType.Q8_0,
                DataTypeMapping.materializedType(GGMLType.Q8_0, ExecutionTarget.GPU));
    }

    @Test
    public void theDeviceMaterializesEverythingElseAsQ8_0() {
        for (GGMLType fileType : DEVICE_MATERIALIZED) {
            assertSame(
                    fileType + " has no device kernel and must be materialized",
                    DataType.Q8_0,
                    DataTypeMapping.materializedType(fileType, ExecutionTarget.GPU));
        }
    }

    /**
     * The two targets disagree exactly where the converting path converts.
     *
     * <p>Restored after an over-eager edit removed it alongside the test beside it. The property is
     * still worth holding: if {@link DataTypeMapping#materializedType}'s target parameter made no
     * difference anywhere, it would be decoration.
     */
    @Test
    public void theTargetsDisagreeExactlyWhereTheDeviceLacksAKernel() {
        for (GGMLType fileType : DEVICE_MATERIALIZED) {
            assertFalse(
                    fileType + ": CPU and the converting device path must not agree here",
                    DataTypeMapping.materializedType(fileType, ExecutionTarget.CPU)
                            == DataTypeMapping.materializedType(fileType, ExecutionTarget.GPU));
        }
        for (GGMLType fileType : new GGMLType[] {GGMLType.F32, GGMLType.F16, GGMLType.Q8_0}) {
            assertSame(
                    fileType + ": both targets execute this one directly",
                    DataTypeMapping.materializedType(fileType, ExecutionTarget.CPU),
                    DataTypeMapping.materializedType(fileType, ExecutionTarget.GPU));
        }
    }

    /**
     * The converting device path promotes every block quantization to Q8_0, and says so.
     *
     * <p>This replaces a test asserting that no target materializes a "format-decoded" type. That
     * framing is gone: the quantizations are kept in the file's own layout by {@code
     * ModelLoader.loadTornadoTensorNative}, and what {@link DataTypeMapping#materializedType} still
     * describes is the older loading path some families remain on. Naming it is the point — the
     * conversion is a declared decision that reaches the memory plan, not a silent one.
     */
    @Test
    public void theConvertingDevicePathPromotesToQ8_0() {
        for (GGMLType fileType :
                new GGMLType[] {
                    GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K
                }) {
            assertEquals(
                    fileType + " on the converting device path",
                    DataType.Q8_0,
                    DataTypeMapping.materializedType(fileType, ExecutionTarget.GPU));
            assertEquals(
                    fileType + " is untouched on the host",
                    DataTypeMapping.sourceType(fileType),
                    DataTypeMapping.materializedType(fileType, ExecutionTarget.CPU));
        }
    }

    @Test
    public void supportIsAskedRatherThanGuessed() {
        for (ExecutionTarget target : ExecutionTarget.values()) {
            assertTrue(DataTypeMapping.isSupported(GGMLType.Q4_K, target));
            assertFalse(DataTypeMapping.isSupported(GGMLType.Q2_K, target));
        }
    }

    @Test
    public void activationsFollowTheWeightsExceptThatFloatsStayFloats() {
        assertSame(DataType.F16, DataTypeMapping.activationType(GGMLType.F16));
        assertSame(DataType.F16, DataTypeMapping.activationType(GGMLType.F32));
        assertSame(DataType.Q8_0, DataTypeMapping.activationType(GGMLType.Q8_0));
        assertSame(DataType.Q8_0, DataTypeMapping.activationType(GGMLType.Q4_0));
        assertSame(DataType.Q8_0, DataTypeMapping.activationType(GGMLType.Q6_K));
    }

    /**
     * The mapping replaces {@code effectiveGpuWeightType}, so it has to return what that returned —
     * otherwise this is a behaviour change wearing a refactor's clothes.
     */
    @Test
    public void theGpuMappingAgreesWithTheLoadersOldAnswer() {
        assertEquals(GGMLType.F16, gpuFileType(GGMLType.F16));
        assertEquals(GGMLType.F32, gpuFileType(GGMLType.F32));
        assertEquals(GGMLType.Q8_0, gpuFileType(GGMLType.Q8_0));
        assertEquals(GGMLType.Q8_0, gpuFileType(GGMLType.Q4_0));
        assertEquals(GGMLType.Q8_0, gpuFileType(GGMLType.Q4_K));
        assertEquals(GGMLType.Q8_0, gpuFileType(GGMLType.Q5_K));
        assertEquals(GGMLType.Q8_0, gpuFileType(GGMLType.Q6_K));
    }

    @Test
    public void theActivationMappingAgreesWithTheQuantizationString() {
        assertEquals("FP16", asStateString(DataTypeMapping.activationType(GGMLType.F16)));
        assertEquals("Q8_0", asStateString(DataTypeMapping.activationType(GGMLType.Q8_0)));
        assertEquals("Q8_0", asStateString(DataTypeMapping.activationType(GGMLType.Q4_0)));
        assertEquals("Q8_0", asStateString(DataTypeMapping.activationType(GGMLType.Q4_K)));
        assertEquals("Q8_0", asStateString(DataTypeMapping.activationType(GGMLType.Q6_K)));
    }

    /** The transitional reverse direction, used while the loaders still speak GGMLType. */
    @Test
    public void everyRuntimeTypeHasAFileTypeWhileTheBridgeExists() {
        for (DataType dataType : DataType.values()) {
            assertSame(dataType, DataTypeMapping.sourceType(DataTypeMapping.asFileType(dataType)));
        }
    }

    private static GGMLType gpuFileType(GGMLType fileType) {
        return DataTypeMapping.asFileType(
                DataTypeMapping.materializedType(fileType, ExecutionTarget.GPU));
    }

    /** How the activation representation is spelled by {@code Configuration.quantization()}. */
    private static String asStateString(DataType activation) {
        return activation == DataType.F16 ? "FP16" : activation.name();
    }
}
