package org.beehive.jitllm.model.loader;

import static org.junit.Assert.assertEquals;

import org.beehive.jitllm.backend.tornado.tensor.FP16TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q4_KTornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.Q8_0TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.Shape;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/** What a load reports for its projections, which picks the plan's layers. */
public class KeptTypeTest {

    private static final Shape SHAPE = Shape.of(32);

    private static TornadoTensor f16() {
        return new FP16TornadoTensor(new HalfFloatArray(32), SHAPE);
    }

    private static TornadoTensor q8_0() {
        return new Q8_0TornadoTensor(new ByteArray(34), SHAPE);
    }

    private static TornadoTensor q4_k() {
        return new Q4_KTornadoTensor(new ByteArray(144), SHAPE);
    }

    @Test
    public void uniformFilesReportTheirType() {
        assertEquals(DataType.F16, ModelLoader.keptType(new TornadoTensor[] {f16(), f16()}));
        assertEquals(DataType.Q8_0, ModelLoader.keptType(new TornadoTensor[] {q8_0(), q8_0()}));
    }

    @Test
    public void aMixedFileReportsTheFirstTypeOtherThanQ8_0() {
        assertEquals(DataType.Q4_K, ModelLoader.keptType(new TornadoTensor[] {q8_0(), q4_k()}));
    }

    /** UD-Q8_K_XL: F16 in some layers, Q8_0 in the rest, runs the Q8_0 plans. */
    @Test
    public void f16BesideQ8_0ReportsQ8_0() {
        assertEquals(
                DataType.Q8_0,
                ModelLoader.keptType(
                        new TornadoTensor[] {f16(), q8_0()}, new TornadoTensor[] {q8_0()}));
    }

    @Test
    public void f16BesideAnotherTypeReportsThatType() {
        assertEquals(
                DataType.Q4_K, ModelLoader.keptType(new TornadoTensor[] {f16(), q4_k(), q8_0()}));
    }
}
