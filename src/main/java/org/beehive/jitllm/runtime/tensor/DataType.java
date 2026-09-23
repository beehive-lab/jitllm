package org.beehive.jitllm.runtime.tensor;

import org.beehive.jitllm.api.Experimental;

/**
 * How values are represented where the engine actually computes with them.
 *
 * <p>This is the runtime's own vocabulary, deliberately smaller than the file format's. GGUF's
 * {@code GGMLType} describes what is in a file; {@code DataType} describes what executes. The two
 * are related by an explicit mapping in the format layer rather than by being the same enum, which
 * is what lets the runtime, program, operation and backend layers stay free of format types (Rule
 * 4).
 *
 * <h2>What is not here</h2>
 *
 * <h2>What this type answers, and what it does not</h2>
 *
 * <p>It describes a <b>representation</b>: how values are laid out, and whether that layout is
 * blocks with scales ({@link #isQuantized()}). That is all. Three questions it deliberately does
 * not answer, because each has a different owner and conflating them is what made the engine
 * believe quantized weights were CPU-only:
 *
 * <ul>
 *   <li><b>Must arithmetic decode a block to read a value?</b> True of every quantized
 *       representation here, on <i>every</i> backend. A device kernel decodes inside its dot
 *       product exactly as a host one does. It is a property of the representation and it implies
 *       nothing about which backend can hold it.
 *   <li><b>Can a backend store it?</b> The backend's storage vocabulary — for TornadoVM, whether a
 *       {@code TornadoTensor} wrapper exists.
 *   <li><b>Does a given operation have a kernel for it?</b> {@code OperationSupport}, per operation
 *       and per target. A matrix-vector product reads all six quantizations; matrix-matrix has
 *       tensor-core kernels for two. "The GPU cannot do Q5_K" was never a fact about Q5_K.
 * </ul>
 */
@Experimental
public enum DataType {

    /** 32-bit float. What the CPU accumulates in, whatever the weights are stored as. */
    F32(false),

    /** 16-bit float. Stored and computed with directly on the GPU path. */
    F16(false),

    /**
     * 16-bit brain float: the same exponent range as {@link #F32} with fewer mantissa bits.
     *
     * <p>Not block-encoded: the CPU materializes a tensor in this representation and reads it
     * directly. The GPU does not execute it, and converts to {@link #F16} at load instead — a
     * narrowing that loses exponent range, which is why the conversion is stated in {@link
     * #materializedFallback()} rather than left implicit. The device type exists (TornadoVM's
     * {@code BFloat16Array}, 5.2.0), so this is today's behaviour and not a permanent limit.
     */
    BF16(false),

    /** 8-bit block quantization: signed 8-bit values with a per-block scale. */
    Q8_0(true),

    /**
     * 4-bit block quantization, 32 values to a block with one scale. Like the K-quants it is
     * decoded during compute on the CPU and materialized as {@link #Q8_0} for the GPU.
     */
    Q4_0(true),

    /**
     * 4-bit block quantization with a per-block minimum as well as a scale: {@code d * q + m}, 32
     * values to a block. Like {@link #Q4_0} it is decoded during compute on the CPU and
     * materialized as {@link #Q8_0} for the GPU.
     *
     * <p>Here because Qwen3.8-27B mixes it into an otherwise Q4_0 file — the first eight layers'
     * {@code ffn_down} tensors of `Qwen3.8-27B-Q4_0.gguf` are Q4_1.
     */
    Q4_1(true),

    /**
     * 4-bit K-quantization. CPU only, decoded during compute; the GPU materializes {@link #Q8_0}.
     */
    Q4_K(true),

    /**
     * 5-bit K-quantization. CPU only, decoded during compute; the GPU materializes {@link #Q8_0}.
     */
    Q5_K(true),

    /**
     * 6-bit K-quantization. CPU only, decoded during compute; the GPU materializes {@link #Q8_0}.
     */
    Q6_K(true);

    private final boolean quantized;

    DataType(boolean quantized) {
        this.quantized = quantized;
    }

    /** Whether values are stored in blocks with scales rather than as plain floats. */
    public boolean isQuantized() {
        return quantized;
    }

    /**
     * The representation this one is <b>narrowed</b> to when no arithmetic exists for it.
     *
     * <p>One case, and it is a genuine narrowing rather than a capability gap: {@link #BF16} loses
     * mantissa bits to {@link #F16} because no BF16 device arithmetic is used. The device type
     * exists (TornadoVM's {@code BFloat16Array}), so this is today's behaviour, not a permanent
     * limit.
     *
     * <p>It used to answer {@link #Q8_0} for every quantized representation, which is how a 4-bit
     * model came to occupy twice its size on a device. It no longer does: a quantized tensor is
     * kept in the layout the file gave it, and an operation with no kernel for that layout is
     * refused by name rather than served by conversion.
     *
     * @return the narrowed representation, or this type when it needs none
     */
    public DataType narrowedFallback() {
        return this == BF16 ? F16 : this;
    }
}
