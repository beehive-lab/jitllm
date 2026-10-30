package org.beehive.jitllm.model.loader;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.beehive.jitllm.backend.tornado.tensor.Q4_0TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensorLoader;
import org.beehive.jitllm.format.GGMLTensorEntry;
import org.beehive.jitllm.format.GGMLType;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.Shape;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/**
 * A device tensor read from a file records the dimensions the file gives it, so that what happens
 * to it on the device can be decided from the tensor itself.
 *
 * <p>No model file and no device: the entries are synthesized with TornadoVM's array header in
 * front, as the loader receives them.
 */
public class DeviceTensorShapeTest {

    /** Two rows of 64 inputs: dimension 0 is the row length, in GGUF {@code ne} order. */
    private static final int[] SHAPE = {64, 2};

    private static final int ELEMENTS = SHAPE[0] * SHAPE[1];

    private static GGMLTensorEntry entry(Arena arena, GGMLType type) {
        long dataBytes = type.byteSizeFor(ELEMENTS);
        long header = TornadoNativeArray.ARRAY_HEADER;
        MemorySegment withHeader = arena.allocate(header + dataBytes);
        return new GGMLTensorEntry(withHeader, "blk.0.ffn_up.weight", type, SHAPE, withHeader);
    }

    private static void assertShape(TornadoTensor tensor) {
        assertArrayEquals(new long[] {64, 2}, tensor.shape().dimensions());
    }

    @Test
    public void aTensorKeptAsTheFileStoresItRecordsItsShape() {
        try (Arena arena = Arena.ofConfined()) {
            TornadoTensor asStored =
                    ModelLoader.loadTornadoTensorNative(entry(arena, GGMLType.Q4_0));
            TornadoTensor f32 = ModelLoader.loadTornadoTensorNative(entry(arena, GGMLType.F32));

            assertEquals(DataType.Q4_0, asStored.dataType());
            assertShape(asStored);
            assertShape(f32);
        }
    }

    /** A tensor computed at load, such as a RoPE table, states its shape like any other. */
    @Test
    public void aComputedTensorStatesItsShape() {
        TornadoTensor table = TornadoTensorLoader.fromFloats(new float[48]);

        assertArrayEquals(new long[] {48}, table.shape().dimensions());
    }

    @Test
    public void everyDeviceTensorStatesItsShape() {
        assertThrows(
                NullPointerException.class,
                () -> new Q4_0TornadoTensor(new ByteArray(18), (Shape) null));
    }
}
