package org.beehive.jllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The one definition of the KV block-table walk. Kernels call it; nothing else does.
 *
 * <p>KV lives in fixed-size blocks. A block holds {@code blockSize} consecutive positions of one
 * sequence <b>across all layers</b>, and a per-slot block table maps a logical block to a physical
 * one, so an index can finally say <i>which lease</i>:
 *
 * <pre>
 * pool[ physBlock * (numLayers * blockSize * kvDim)   // blockStride
 *     + layer     * (blockSize * kvDim)               // layerOff
 *     + (pos % blockSize) * kvDim
 *     + c ]
 * </pre>
 *
 * <p>This replaces {@code layer * contextLength * kvDim + pos * kvDim}, which is per-sequence
 * contiguous and has nowhere to put a lease term. The shape is #129's, already proven on the paged
 * batch-decode path, so the single-token and batch paths converge rather than growing two
 * addressing schemes.
 *
 * <p><b>{@code slot} is an address, not an identity.</b> It indexes the block table the kernel is
 * bound to. A lease's identity and generation are validity concerns, checked on the host;
 * conflating them with the slot would let a stale lease read someone else's KV instead of being
 * refused.
 *
 * <p>The runtime {@code /} and {@code %} are deliberate: {@code blockSize} is not a compile-time
 * constant, so they do not lower to shifts and masks. The first measurement is meant to be the
 * honest cost of the naive walk. The tiled kernels could hoist the lookup per tile — that is held
 * in reserve for if the measurement asks for it, not added ahead of it.
 */
final class KvBlockAddress {

    private KvBlockAddress() {}

    /** {@code blockSize} out of the packed config {@code blockSize | (maxBlocksPerSlot << 16)}. */
    static int blockSize(int blockCfg) {
        return blockCfg & 0xFFFF;
    }

    /**
     * {@code maxBlocksPerSlot} out of the packed config. Packed to stay inside the task arg limit.
     */
    static int maxBlocksPerSlot(int blockCfg) {
        return blockCfg >>> 16;
    }

    /** {@code layer * (blockSize * kvDim)}, the layer's offset within a block. */
    static int layerOffset(int layer, int kvDim, int blockCfg) {
        return layer * ((blockCfg & 0xFFFF) * kvDim);
    }

    /**
     * Element offset of position {@code pos} for the lease at {@code slot}, in the layer whose
     * offset within a block is {@code layerOff}.
     */
    static int offset(
            IntArray blockTable,
            int slot,
            int pos,
            int layerOff,
            int kvDim,
            int blockCfg,
            int blockStride) {
        int blockSize = blockCfg & 0xFFFF;
        int physBlock = blockTable.get(slot * (blockCfg >>> 16) + pos / blockSize);
        return physBlock * blockStride + layerOff + (pos % blockSize) * kvDim;
    }
}
