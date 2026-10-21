package org.beehive.jitllm.backend.tornado.scheduling;

import org.beehive.jitllm.backend.tornado.device.TornadoDevices;
import org.beehive.jitllm.runtime.backend.BackendId;

// @formatter:off
/**
 * Whether the batched-prefill projections run as tiled matrix multiplications.
 *
 * <p>The kernels they replace launch one workgroup per (token, output row), so every token reads
 * the whole weight matrix again and reduces its row through a shared-memory tree. The tiled kernels
 * compute 64-row by 64-token tiles from weights and activations staged in threadgroup memory,
 * reading each weight once per 64 tokens. Measured on an M4 Pro, Qwen3-0.6B F16 prefill with
 * lane-cooperative attention, pp512 at batch 128: 404 tok/s with the per-row kernels, 1590 tiled
 * (1962 at batch 512); greedy text identical to single-token prefill. Metal only: the other
 * backends run the tensor-core kernels or were not measured with these.
 */
// @formatter:on
public final class BatchPrefillGemmPolicy {

    private BatchPrefillGemmPolicy() {}

    /** Whether the active device runs the tiled batched-prefill projections. */
    public static boolean tiled() {
        return TornadoDevices.current().backend().equals(BackendId.METAL);
    }

    // @formatter:off
    /**
     * Whether the tiled projections run on SIMD-group matrices ({@link
     * org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillSimdgroupKernels}) instead
     * of scalar FMAs. On by default with {@link #tiled()}; {@code
     * -Djitllm.metal.simdgroupGemm=false} selects the scalar kernels.
     *
     * <p>Measured on an M4 Pro, Qwen3-0.6B, batch 256 (pp512 / pp2048 tok/s): F16 1965 / 1175
     * scalar, 2929 / 1461 SIMD-group; Q8_0 1944 / 1166 scalar, 2931 / 1462 SIMD-group. Greedy text
     * identical; the CPU/GPU parity tests pass. Q8_0 weights are staged as {@code float}: rounding
     * the dequantized weights to {@code half} was faster but failed the batched-prefill parity
     * budget. FP16 stages through two alternating buffers (one barrier per k-slice): kernel time at
     * batch 256 fell 11% for gate/up and the down projection and 5% for QKV. Reading the staged
     * tiles with vector-width loads ({@code getFloat4} / {@code getHalf4}) took Qwen3-0.6B pp512 /
     * pp2048 to 4171 / 2886 tok/s for F16 and 3802 / 2693 for Q8_0.
     */
    // @formatter:on
    public static boolean simdgroup() {
        return tiled() && !"false".equals(System.getProperty("jitllm.metal.simdgroupGemm"));
    }

    // @formatter:off
    /**
     * Whether batched-prefill attention over the FP32 cache runs on SIMD-group matrices ({@link
     * org.beehive.jitllm.backend.tornado.kernels.TransformerPagedKvBatchPrefillSimdgroupKernels})
     * where the lane-cooperative kernel would otherwise run. On by default with {@link #tiled()};
     * {@code -Djitllm.metal.simdgroupAttention=false} selects the lane-cooperative kernel.
     *
     * <p>Measured on an M4 Pro, Qwen3-0.6B, batch 256, SIMD-group projections (pp512 / pp2048
     * tok/s): F16 2913 / 1452 lane-cooperative, 3698 / 2645 SIMD-group; Q8_0 2912 / 1453
     * lane-cooperative, 3664 / 2636 SIMD-group. Greedy text identical; the CPU/GPU parity tests
     * pass.
     */
    // @formatter:on
    public static boolean simdgroupAttention() {
        return tiled() && !"false".equals(System.getProperty("jitllm.metal.simdgroupAttention"));
    }

    /**
     * Whether {@link #simdgroupAttention()} holds and a SIMD-group kernel exists for this head
     * width.
     */
    public static boolean simdgroupAttention(int headSize) {
        return simdgroupAttention() && (headSize == 64 || headSize == 96 || headSize == 128);
    }

    /**
     * Whether the SIMD-group projections can run a model of these sizes: their double-buffered
     * k-loop needs every contraction length, and their row tiles every projection height, to be a
     * multiple of 64.
     */
    public static boolean simdgroup(int dim, int qDim, int kvDim, int hiddenDim) {
        return simdgroup()
                && dim % 64 == 0
                && qDim % 64 == 0
                && kvDim % 64 == 0
                && hiddenDim % 64 == 0;
    }
}
