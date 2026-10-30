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
 * What a file's tensor becomes on each target. Both keep every block quantization as the file
 * stores it; they differ only on BF16, which the device holds as F16.
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

    private static final GGMLType[] BLOCK_QUANTIZED = {
        GGMLType.Q8_0, GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K
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
    public void theDeviceKeepsEveryBlockQuantizationAsTheFileStoresIt() {
        for (GGMLType fileType : BLOCK_QUANTIZED) {
            assertSame(
                    fileType + " is kept on the device",
                    DataTypeMapping.sourceType(fileType),
                    DataTypeMapping.materializedType(fileType, ExecutionTarget.GPU));
        }
    }

    /** The targets disagree on BF16 alone: the device has no BF16 kernels and holds it as F16. */
    @Test
    public void theTargetsDisagreeOnlyOnBf16() {
        assertSame(
                DataType.BF16,
                DataTypeMapping.materializedType(GGMLType.BF16, ExecutionTarget.CPU));
        assertSame(
                DataType.F16, DataTypeMapping.materializedType(GGMLType.BF16, ExecutionTarget.GPU));
        for (GGMLType fileType : RECOGNIZED) {
            assertSame(
                    fileType + ": both targets hold this one as it is",
                    DataTypeMapping.materializedType(fileType, ExecutionTarget.CPU),
                    DataTypeMapping.materializedType(fileType, ExecutionTarget.GPU));
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

    /** How the activation representation is spelled by {@code Configuration.quantization()}. */
    private static String asStateString(DataType activation) {
        return activation == DataType.F16 ? "FP16" : activation.name();
    }
}
