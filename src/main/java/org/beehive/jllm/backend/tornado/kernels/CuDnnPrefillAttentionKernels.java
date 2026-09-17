package org.beehive.jllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The layout adapters around a cuDNN scaled-dot-product-attention call on the batched
 * prefill path.
 *
 * <p>cuDNN's binding takes three contiguous {@code [batch][head][seq][headDim]} FP16
 * tensors and one head count. jllm's prefill keeps Q inside a packed FP32 QKV buffer and
 * its K/V in a paged FP16 cache with a block table, and serves several query heads from
 * one KV head. These three kernels bridge exactly that gap and nothing else: the
 * projections, Q/K normalization, RoPE and the output projection are untouched, and the
 * KV cache is still written by the existing RoPE kernel — this only reads it.
 *
 * <p>Applicable to the first prefill chunk of a sequence only, because cuDNN's causal mask
 * aligns query <i>i</i> to key <i>i</i>. That is the correct mask when the query block is
 * the whole prefix (query length == key length) and the wrong one for any later chunk,
 * where the queries sit at an offset into a longer key range. The caller enforces this.
 */
public final class CuDnnPrefillAttentionKernels {

    private CuDnnPrefillAttentionKernels() {}

    /**
     * Packed FP32 QKV {@code [tok][qkvStride]} to FP16 {@code [head][tok][headDim]}.
     *
     * <p>Padding rows are zero-filled rather than skipped: they still take part in the
     * attention the library computes, so leaving them undefined would feed NaNs into the
     * softmax of the real rows' key range.
     */
    public static void packQ(
            IntArray batchStartPosHolder,
            FloatArray qkvBatch,
            HalfFloatArray qOut,
            int headSize,
            int batchSize,
            int qkvStride) {
        for (@Parallel int i = 0; i < qOut.getSize(); i++) {
            int d = i % headSize;
            int tok = (i / headSize) % batchSize;
            int h = i / (headSize * batchSize);
            if (tok < batchStartPosHolder.get(1)) {
                qOut.set(i, new HalfFloat(qkvBatch.get(tok * qkvStride + h * headSize + d)));
            } else {
                qOut.set(i, new HalfFloat(0.0f));
            }
        }
    }

    /**
     * Paged FP16 K/V to contiguous FP16 {@code [head][tok][headDim]}, expanding grouped
     * query attention: query head {@code h} reads KV head {@code h / kvMul}.
     */
    public static void gatherKvExpanded(
            IntArray batchStartPosHolder,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            IntArray blockTable,
            HalfFloatArray kOut,
            HalfFloatArray vOut,
            int headSize,
            int batchSize,
            int kvDim,
            int kvMul,
            int layerIndex,
            int blockCfg,
            int blockStride) {
        for (@Parallel int i = 0; i < kOut.getSize(); i++) {
            int d = i % headSize;
            int tok = (i / headSize) % batchSize;
            int h = i / (headSize * batchSize);
            if (tok < batchStartPosHolder.get(1)) {
                int pos = batchStartPosHolder.get(0) + tok;
                int slot = batchStartPosHolder.get(2);
                int layerOff = KvBlockAddress.layerOffset(layerIndex, kvDim, blockCfg);
                int base =
                        KvBlockAddress.offset(
                                blockTable, slot, pos, layerOff, kvDim, blockCfg, blockStride);
                int src = base + (h / kvMul) * headSize + d;
                kOut.set(i, keyCache.get(src));
                vOut.set(i, valueCache.get(src));
            } else {
                kOut.set(i, new HalfFloat(0.0f));
                vOut.set(i, new HalfFloat(0.0f));
            }
        }
    }

    /** cuDNN output {@code [head][tok][headDim]} back to {@code attnOutFP16[tok][qDim]}. */
    public static void scatterAttnOut(
            HalfFloatArray sdpaOut, HalfFloatArray attnOutFP16, int headSize, int batchSize, int qDim) {
        for (@Parallel int i = 0; i < sdpaOut.getSize(); i++) {
            int d = i % headSize;
            int tok = (i / headSize) % batchSize;
            int h = i / (headSize * batchSize);
            attnOutFP16.set(tok * qDim + h * headSize + d, sdpaOut.get(i));
        }
    }
}
