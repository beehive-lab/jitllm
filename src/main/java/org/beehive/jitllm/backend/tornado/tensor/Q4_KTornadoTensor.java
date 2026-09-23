package org.beehive.jitllm.backend.tornado.tensor;

import java.lang.foreign.MemorySegment;
import org.beehive.jitllm.format.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

/**
 * A quantized tensor in the {@link GGMLType#Q4_K} format, <b>retained</b> on the device.
 *
 * <p>Backed by the file's own bytes, exactly as {@link Q8_0TornadoTensor} is: the segment is
 * wrapped, not converted, so nothing is materialized at load. Each super-block covers 256 weights
 * in 144 bytes — 4.5 bits per weight against Q8_0's 8.5:
 *
 * <pre>
 *   offset  0   d       (fp16)   super-block scale for the quantized scales
 *   offset  2   dmin    (fp16)   super-block scale for the quantized mins
 *   offset  4   scales  (12 B)   eight 6-bit scale/min pairs, packed
 *   offset 16   qs     (128 B)   256 4-bit weights, low nibble then high nibble
 * </pre>
 *
 * <p>A weight is {@code d * scale(sub) * q - dmin * min(sub)}, decoded inside the dot product by
 * {@link org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_K} — the same shape
 * the Q8_0 kernels already use, which is why this needs no new plumbing: the layers pass {@link
 * #asByteArray()} just as they do for Q8_0, and only the kernel that reads it differs.
 *
 * <p><b>Why this exists</b>: the GPU used to materialize Q4_K as Q8_0 at load, which nearly doubled
 * its footprint — a 13.7 GiB Devstral needed ~24 GiB of device memory, and exhausted a 24 GiB
 * machine before it could run at all. Retaining the file's own representation is what makes that
 * model fit.
 */
public class Q4_KTornadoTensor extends TornadoTensor {

    /** Weights per super-block. */
    public static final int QK_K = 256;

    /** Bytes per super-block: 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs). */
    public static final int BLOCK_BYTES = 144;

    /** Byte offset of the packed 6-bit scale/min pairs within a super-block. */
    public static final int SCALES_OFFSET = 4;

    /** Byte offset of the 4-bit weights within a super-block. */
    public static final int QS_OFFSET = 16;

    private final ByteArray tornadoNativeArray;

    public Q4_KTornadoTensor(ByteArray byteArray) {
        this.tornadoNativeArray = byteArray;
    }

    public static Q4_KTornadoTensor fromTornadoMemorySegment(MemorySegment segment) {
        return new Q4_KTornadoTensor(ByteArray.fromSegmentShallow(segment));
    }

    @Override
    public ByteArray asByteArray() {
        return tornadoNativeArray;
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q4_K;
    }
}
