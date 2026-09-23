package org.beehive.jllm.runtime.tensor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.lang.foreign.MemorySegment;
import java.lang.reflect.Method;
import org.beehive.jllm.backend.tornado.tensor.Q8_0TornadoTensor;
import org.beehive.jllm.format.GGMLType;
import org.beehive.jllm.tensor.standard.Q8_0FloatTensor;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

/**
 * The pair matters more than either half: a {@code dataType()} that disagreed with {@code type()}
 * would be worse than none at all, and a deprecation nobody can see is not a migration signal.
 */
public class DataTypeAccessorsTest {

    @Test
    public void aHostTensorReportsItsRuntimeRepresentation() {
        Q8_0FloatTensor tensor = new Q8_0FloatTensor(32, MemorySegment.NULL);
        assertSame(DataType.Q8_0, tensor.dataType());
    }

    @Test
    public void aDeviceTensorReportsItsRuntimeRepresentation() {
        Q8_0TornadoTensor tensor = new Q8_0TornadoTensor(new ByteArray(34));
        assertSame(DataType.Q8_0, tensor.dataType());
        assertEquals("the old accessor still answers, and agrees", GGMLType.Q8_0, tensor.type());
    }

    @Test
    public void theFileTypeAccessorsThatRemainAreDeprecated() throws Exception {
        assertDeprecated(org.beehive.jllm.backend.tornado.tensor.TornadoTensor.class, "type");
        assertDeprecated(org.beehive.jllm.model.Configuration.class, "quantization");
    }

    private static void assertDeprecated(Class<?> type, String methodName) throws Exception {
        Method method = type.getDeclaredMethod(methodName);
        assertTrue(
                type.getSimpleName() + "." + methodName + " must be deprecated, not removed",
                method.isAnnotationPresent(Deprecated.class));
    }

    /**
     * Activations are a different question from weights, and the accessor answers the one it is
     * named for: a K-quant model has Q8_0 activations and K-quant weights.
     */
    @Test
    public void theConfigurationReportsItsActivationRepresentation() {
        assertSame(DataType.F16, configurationReporting("FP16").activationType());
        assertSame(DataType.Q8_0, configurationReporting("Q8_0").activationType());
    }

    private static org.beehive.jllm.model.Configuration configurationReporting(
            String quantization) {
        return new org.beehive.jllm.model.Configuration() {
            @Override
            public String quantization() {
                return quantization;
            }

            @Override
            public int dim() {
                return 0;
            }

            @Override
            public int hiddenDim() {
                return 0;
            }

            @Override
            public int numberOfLayers() {
                return 0;
            }

            @Override
            public int numberOfHeads() {
                return 0;
            }

            @Override
            public int numberOfKeyValueHeads() {
                return 0;
            }

            @Override
            public int numberOfHeadsKey() {
                return 0;
            }

            @Override
            public int vocabularySize() {
                return 0;
            }

            @Override
            public int contextLength() {
                return 0;
            }

            @Override
            public int contextLengthModel() {
                return 0;
            }

            @Override
            public float rmsNormEps() {
                return 0;
            }

            @Override
            public float ropeTheta() {
                return 0;
            }

            @Override
            public int headSize() {
                return 0;
            }

            @Override
            public int kvDim() {
                return 0;
            }

            @Override
            public int kvMul() {
                return 0;
            }
        };
    }
}
