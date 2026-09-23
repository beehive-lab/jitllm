package org.beehive.jllm.backend.tornado.tensor;

import java.lang.foreign.MemorySegment;
import org.beehive.jllm.format.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

/**
 * A quantized tensor in the {@link GGMLType#Q6_K} format, <b>retained</b> on the device.
 *
 * <p>Backed by the file's own bytes, exactly as {@link Q8_0TornadoTensor} is. Each super-block
 * covers 256 weights in 210 bytes — 6.5625 bits per weight against Q8_0's 8.5:
 *
 * <pre>
 *   offset   0   ql     (128 B)   the low 4 bits of each 6-bit weight
 *   offset 128   qh      (64 B)   the high 2 bits, four weights to a byte
 *   offset 192   scales  (16 B)   one signed scale per 16 weights
 *   offset 208   d       (fp16)   super-block scale
 * </pre>
 *
 * <p>Decoded inside the dot product by {@link
 * org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K}.
 *
 * <p><b>Why this exists</b>: retaining Q4_K alone was not enough, because a "Q4_K_M" file mixes
 * formats per tensor and per layer — Devstral holds attn_v and ffn_down as Q6_K in half its layers.
 * Those would still have been materialized as Q8_0, so the memory the amendment set out to save was
 * only partly saved.
 */
public class Q6_KTornadoTensor extends TornadoTensor {

    /** Weights per super-block. */
    public static final int QK_K = 256;

    /** Bytes per super-block: 128 (ql) + 64 (qh) + 16 (scales) + 2 (d). */
    public static final int BLOCK_BYTES = 210;

    private final ByteArray tornadoNativeArray;

    public Q6_KTornadoTensor(ByteArray byteArray) {
        this.tornadoNativeArray = byteArray;
    }

    public static Q6_KTornadoTensor fromTornadoMemorySegment(MemorySegment segment) {
        return new Q6_KTornadoTensor(ByteArray.fromSegmentShallow(segment));
    }

    @Override
    public ByteArray asByteArray() {
        return tornadoNativeArray;
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q6_K;
    }
}
