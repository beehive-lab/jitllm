package org.beehive.jllm.backend.tornado.tensor;

import org.beehive.jllm.format.DataTypeMapping;
import org.beehive.jllm.format.GGMLType;
import org.beehive.jllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.Int8Array;

/**
 * Base class for TornadoVM-compatible tensor types. These tensors wrap TornadoVM native arrays for
 * GPU execution.
 */
public abstract class TornadoTensor {

    /**
     * @deprecated Use {@link #dataType()}. A materialized device tensor is not a file type — a
     *     K-quant or Q4_0 file arrives here as {@link DataType#Q8_0}, and saying so in the file's
     *     vocabulary invites the two to be confused (Rule 4).
     */
    @Deprecated
    public abstract GGMLType type();

    /** The representation this device tensor holds, in the runtime's vocabulary. */
    public DataType dataType() {
        return DataTypeMapping.sourceType(type());
    }

    /**
     * Get as FloatArray (for F32 tensors).
     *
     * @throws UnsupportedOperationException if not F32
     */
    public FloatArray asFloatArray() {
        throw new UnsupportedOperationException(
                "Not a FloatArray tensor: " + this.getClass().getSimpleName());
    }

    /**
     * Get as HalfFloatArray (for F16 tensors).
     *
     * @throws UnsupportedOperationException if not F16
     */
    public HalfFloatArray asHalfFloatArray() {
        throw new UnsupportedOperationException(
                "Not a HalfFloatArray tensor: " + this.getClass().getSimpleName());
    }

    /**
     * Get as ByteArray (for Q8_0 tensors).
     *
     * @throws UnsupportedOperationException if not Q8_0
     */
    public ByteArray asByteArray() {
        throw new UnsupportedOperationException(
                "Not a Q8_0 ByteArray tensor: " + this.getClass().getSimpleName());
    }

    /**
     * Get quantized scales (for Q8_0 tensors).
     *
     * @throws UnsupportedOperationException if not quantized
     */
    public HalfFloatArray getScales() {
        throw new UnsupportedOperationException(
                "Not a quantized tensor: " + this.getClass().getSimpleName());
    }

    /**
     * Get quantized values (for Q8_0 tensors).
     *
     * @throws UnsupportedOperationException if not quantized
     */
    public Int8Array getQuants() {
        throw new UnsupportedOperationException(
                "Not a quantized tensor: " + this.getClass().getSimpleName());
    }
}
