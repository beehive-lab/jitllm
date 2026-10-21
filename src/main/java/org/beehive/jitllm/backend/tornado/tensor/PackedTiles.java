package org.beehive.jitllm.backend.tornado.tensor;

/**
 * The tile layout a packed Q8_0 or Q4_0 projection is held in on the device: {@code rows} outputs
 * of {@code cols} inputs, in 128-row by 64-input tiles stored in the order the int8 prefill GEMM
 * and the packed decode kernels read them. The host keeps the file's bytes; the device copy is
 * repacked into this layout after upload (see {@code PackedRepack}).
 *
 * @param rows the projection's output rows, a multiple of 128
 * @param cols the projection's inputs, a multiple of 64
 */
public record PackedTiles(int rows, int cols) {

    /** Whether a {@code rows x cols} projection can be packed. */
    public static boolean fits(long rows, long cols) {
        return rows % 128 == 0 && cols % 64 == 0;
    }
}
