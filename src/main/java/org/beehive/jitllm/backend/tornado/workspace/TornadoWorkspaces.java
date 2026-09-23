package org.beehive.jitllm.backend.tornado.workspace;

import org.beehive.jitllm.backend.tornado.kv.TornadoKvStore;
import org.beehive.jitllm.runtime.kv.KvLease;
import org.beehive.jitllm.runtime.kv.KvStorage;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Allocates the device arrays a workspace is made of.
 *
 * <p>Deliberately small and concrete. Four allocators and two shape-specific helpers, named for
 * what they make. Not a builder DSL, not a property bag, and not a generic allocator interface: the
 * families call it directly, and the sizes they pass are the sizes they passed before.
 */
public final class TornadoWorkspaces {

    private TornadoWorkspaces() {}

    public static FloatArray floats(int size) {
        return new FloatArray(size);
    }

    public static HalfFloatArray halfFloats(int size) {
        return new HalfFloatArray(size);
    }

    public static IntArray ints(int size) {
        return new IntArray(size);
    }

    public static ByteArray bytes(int size) {
        return new ByteArray(size);
    }

    /**
     * Writes the decode position, keeping the key/value slot alongside it.
     *
     * <p>The two live in one control array, and a write that sets the position and forgets the slot
     * points the kernels at slot 0 — another session's key/value once storage is shared. Keeping
     * them in one method is what stops that being possible.
     */
    public static void setPosition(TornadoWorkspace workspace, int position, int kvSlot) {
        workspace.positionHolder.set(0, position);
        if (workspace.positionHolder.getSize() > 1) {
            workspace.positionHolder.set(1, kvSlot);
        }
    }

    /** Clears the position for a fresh execution, preserving the slot. */
    public static void resetPosition(TornadoWorkspace workspace, int kvSlot) {
        workspace.positionHolder.init(0);
        if (workspace.positionHolder.getSize() > 1) {
            workspace.positionHolder.set(1, kvSlot);
        }
    }

    /** Writes the slot into an already-allocated control array, if it has room for one. */
    public static void writeSlot(TornadoWorkspace workspace, int kvSlot) {
        if (workspace.positionHolder != null && workspace.positionHolder.getSize() > 1) {
            workspace.positionHolder.set(1, kvSlot);
        }
    }

    /** An identity-mapped block table: logical block i is physical block i. */
    public static void identityBlockTable(TornadoWorkspace workspace, int blocks) {
        workspace.wrapBlockTable = new IntArray(blocks);
        for (int b = 0; b < blocks; b++) {
            workspace.wrapBlockTable.set(b, b);
        }
    }

    /** A session's own zeroed FP32 key/value pair. */
    public static void privateKeyValueFP32(TornadoWorkspace workspace, int elements) {
        workspace.wrapKeyCache = new FloatArray(elements);
        workspace.wrapValueCache = new FloatArray(elements);
        workspace.wrapKeyCache.init(0.f);
        workspace.wrapValueCache.init(0.f);
    }

    /**
     * A zeroed float array.
     *
     * <p>Here rather than {@code floats(n).init(0)} at the call site for the reason every allocator
     * in this class is here: naming a TornadoVM type outside the backend is Rule 1, and its
     * allowlist is empty. A family says how large; the backend says what with.
     */
    public static FloatArray zeroedFloats(int size) {
        FloatArray array = new FloatArray(size);
        array.init(0.f);
        return array;
    }

    /**
     * Zeroes a family's recurrent state, if it has any.
     *
     * <p>For {@code qwen35}, whose convolution windows and delta-net matrices are updated in place
     * and have no position to rewind: a reset must clear them or the next sequence continues the
     * last one. Null-safe, because the same state serves a host-only session that allocated none.
     */
    public static void zeroRecurrentState(TornadoWorkspace workspace) {
        if (workspace.wrapConvState != null) {
            workspace.wrapConvState.init(0.f);
        }
        if (workspace.wrapDeltaState != null) {
            workspace.wrapDeltaState.init(0.f);
        }
    }

    /** Zeroes an already-allocated key/value pair — the families that re-zero after binding. */
    public static void zeroKeyValue(TornadoWorkspace workspace) {
        workspace.wrapKeyCache.init(0.f);
        workspace.wrapValueCache.init(0.f);
    }

    /** Sets the active batch size an MoE plan reads. */
    public static void activeBatchSize(TornadoWorkspace workspace, int size) {
        workspace.activeBatchSizeHolder.init(size);
    }

    /**
     * A zeroed half-precision array. The zero is a {@code HalfFloat}, which is backend vocabulary.
     */
    public static HalfFloatArray zeroedHalfFloats(int size) {
        HalfFloatArray array = new HalfFloatArray(size);
        array.init(new uk.ac.manchester.tornado.api.types.HalfFloat(0.f));
        return array;
    }

    /** The staging activation for an FP16 model. */
    public static void activationFP16(TornadoWorkspace workspace, int size) {
        workspace.embeddingX = new HalfFloatArray(size);
    }

    /** The staging activation for a Q8_0 model: 2 bytes of scale plus 32 quants per block. */
    public static void activationQ8_0(TornadoWorkspace workspace, int size) {
        int blockSize = 32;
        int q8BlockBytes = 34;
        int blocksNeeded = (size + blockSize - 1) / blockSize;
        workspace.embeddingX = new ByteArray(blocksNeeded * q8BlockBytes);
    }

    /**
     * Binds leased key/value storage into a workspace, or reports that there is none to bind.
     *
     * <p>This replaces the {@code KvFieldBinder} seam. That existed because {@code State} could not
     * name {@code TornadoKvStore} and something had to resolve a lease to arrays without putting a
     * cast outside the backend. Inside the backend the cast needs no ceremony, so the seam, its
     * resolver and its service registration are gone.
     *
     * @return whether leased storage was bound; {@code false} means the caller allocates its own,
     *     which is what a CPU-only runtime and an unleased state both do
     */
    public static boolean bindLeasedKeyValue(TornadoWorkspace workspace, KvLease lease) {
        return bindLeasedKeyValue(workspace, lease, null);
    }

    /**
     * As above, also reporting the store's block layout.
     *
     * <p>The layout is <b>the store's</b>, not something the state recomputes: block size, blocks
     * per slot and the stride between blocks are what the pool was built with, and a state that
     * derived its own would address a pool laid out differently. The two values reach the host side
     * through {@code layout}, which is why they are an output parameter rather than a field here —
     * they are read by the key/value addressing on both paths.
     */
    public static boolean bindLeasedKeyValue(
            TornadoWorkspace workspace, KvLease lease, int[] layout) {
        KvStorage storage = lease != null ? lease.storage() : null;
        if (storage == null) {
            return false;
        }
        if (!(storage instanceof TornadoKvStore store)) {
            throw new IllegalStateException(
                    "no compatible backend binding exists for key/value"
                            + " storage "
                            + storage.getClass().getName()
                            + "; this backend binds "
                            + TornadoKvStore.class.getName()
                            + ". A backend/storage mismatch is a configuration error, not a capacity one");
        }
        workspace.wrapKeyCache = store.keyPool();
        workspace.wrapValueCache = store.valuePool();
        workspace.wrapKeyCacheFP16 = store.keyPoolFP16();
        workspace.wrapValueCacheFP16 = store.valuePoolFP16();
        workspace.wrapBlockTable = store.blockTable();
        if (layout != null) {
            layout[0] = store.blockCfg();
            layout[1] = store.blockStride();
        }
        return true;
    }
}
