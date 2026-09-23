package org.beehive.jllm.inference.state;

import org.beehive.jllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.tensor.standard.FloatTensor;

/**
 * Represents the base state structure used during LLM inference. This class provides a common
 * foundation for handling state-related data and functionalities that can be extended by
 * model-specific implementations.
 *
 * <p><b>Key Responsibilities:</b>
 *
 * <ul>
 *   <li>Defines core structures to store and access model state data required for computation.
 *   <li>Can be extended by model-specific state classes (e.g., {@link LlamaState}, {@link
 *       Qwen3State}).
 * </ul>
 *
 * <p><b>Usage:</b> Extend `State` to implement model-specific state configurations while reusing
 * the common structure and functionality provided by this class.
 *
 * <p><b>Note:</b> This class is designed to be generic and does not include any model-specific
 * behavior or fields. Those should be implemented in subclasses.
 */
public abstract class State {

    /**
     * When set ({@code -Djllm.kvcache.fp16=true}), model states that support it additionally
     * allocate half-precision KV caches, and the NVIDIA decode path reads/writes those instead of
     * the FP32 ones (halving KV bandwidth; accumulation stays FP32).
     *
     * <p><b>Read where the arrays are allocated, and where a pool is sized — nowhere else.</b> The
     * key/value representation is <i>storage</i>, not policy, so it does not live in {@code
     * ExecutionPolicy}; and a consumer that wants to know what a state actually holds asks {@link
     * #usesFp16KeyValueCache()} rather than re-deriving it from a property. The two answers are not
     * always the same: a leased state holds whatever the pool was built with.
     */
    @Deprecated public static final boolean USE_FP16_KV = Boolean.getBoolean("jllm.kvcache.fp16");

    /**
     * How this state's key/value storage is shaped, resolved <b>per construction</b>.
     *
     * <p>Defaulted from the properties so the CLI, the benchmark harness and every existing test
     * behave as before. The facade resolves it at model load and hands it in.
     */
    private final org.beehive.jllm.runtime.policy.StorageOptions storageOptions;

    /** How this state was told to store key/value entries. */
    public org.beehive.jllm.runtime.policy.StorageOptions storageOptions() {
        return storageOptions;
    }

    /**
     * Number of KV splits per head for split-KV (flash-decoding) decode attention.
     *
     * <p>Read here and nowhere else. It was declared twice — once in {@code LlamaState}, once in
     * {@code Qwen3State} — reading the same system property with the same default, so a change to
     * one would have silently disagreed with the other.
     */
    public static final int SPLIT_KV = Integer.getInteger("jllm.attention.splitKv.count", 8);

    /**
     * Split-KV attention scratch, or {@code null} for a family that does not use it.
     *
     * <p>Per (head, split): a partial numerator of {@code headSize} plus the block max and block
     * sum. Sized exactly to the configured split count so the split-KV kernel writes every element
     * each layer — a partially written larger buffer is not reliably synced between the split and
     * the combine task. See {@code processHeadsFlashAttentionSplitKVPaged} and {@code
     * combineSplitKVAttention}.
     *
     * <p>The <b>size</b> is family-specific (Qwen3 uses its value-head dimension where Llama uses
     * {@code headSize}), which is why families allocate it; having the field twice was not.
     */

    // current wave of activations
    public final FloatTensor x; // activation at current time stamp (dim,)

    public final FloatTensor xb; // same, but inside a residual branch (dim,)
    public final FloatTensor xb2; // an additional buffer just for convenience (dim,)
    public final FloatTensor hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    public final FloatTensor hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    public final FloatTensor q; // query (dim,)
    public final FloatTensor k; // key (dim,)
    public final FloatTensor v; // value (dim,)
    public final FloatTensor att; // buffer for scores/attention values (n_heads, seq_len)
    public final FloatTensor logits; // output logits
    public final int batchsize;

    /** The prefill batch width this state's workspace was sized for; 1 when it holds none. */
    protected final int prefillBatchWidth;

    /** The device arrays this session executes against, or {@code null} on the host-only path. */
    public final org.beehive.jllm.backend.tornado.workspace.TornadoWorkspace workspace;

    // kv cache
    public final FloatTensor[] keyCache; // (n_layer, seq_len, kv_dim)
    public final FloatTensor[] valueCache; // (n_layer, seq_len, kv_dim)

    // Wrappers for TornadoVM compatibility (FloatArray data structure for TornadoVM acceleration)
    // TornadoVM uses FloatArray for more efficient handling of data, particularly when running on
    // GPU or other accelerators.

    /**
     * The block table every KV kernel walks. Never null.
     *
     * <p>Private to this state and sized for one sequence when the state allocates its own KV, in
     * which case the mapping is the identity and the slot is 0; the shared {@code TornadoKvStore}'s
     * table when the state was built against a lease.
     */

    /** Packed {@code blockSize | (maxBlocksPerSlot << 16)}, as #129 packs it. */
    public final int kvBlockCfg;

    /** {@code numLayers * blockSize * kvDim} — the distance between two blocks. */
    public final int kvBlockStride;

    /**
     * Tokens per KV block: 16, the block size #129 implemented and {@code DelegatingModel} sizes
     * the manager with. Fixed for a pool's lifetime [closed for the number only].
     */
    public static final int KV_BLOCK_SIZE = 16;

    /**
     * This state's slot in the block table, and the value that travels in {@code
     * workspace.positionHolder[1]} and {@code workspace.batchStartPosHolder[2]}.
     *
     * <p>0 while the table is this state's own — one sequence, one slot. When the state is built
     * against a leased {@link org.beehive.jllm.runtime.kv.KvStorage} it is the lease's slot, and
     * several states then address disjoint ranges of one shared table.
     */
    public final int kvSlot;

    /**
     * The lease whose storage this state addresses, or {@code null} when the state owns its own KV
     * arrays. Held so the state can be asked what it is bound to; the storage itself is reached
     * through the wrappers above, resolved once here rather than per token.
     */
    public final org.beehive.jllm.runtime.kv.KvLease kvLease;

    // On-device greedy sampling: the GPU argmax kernel writes the sampled token id here
    // (element 0), so only 1 int crosses to the host instead of the full vocab logits row.

    // store inter
    public int localSize;
    public int
            latestToken; // Keeps track of the most recent token processed by the model. Useful for

    // stateful or autoregressive models.

    // Batch-prefill buffers (allocated when jllm.prefillBatchSize > 1)

    /**
     * The storage options the next state built on this thread should use.
     *
     * <p>A construction-scoped hand-off rather than a constructor parameter, and the reason is
     * mechanical: {@code createStateFields} is called <b>from the {@code State} constructor</b> and
     * overridden by ten subclasses, so a parameter would have to be threaded through every one of
     * their constructors and every {@code Model.createNewState} overload before a single array
     * could be allocated differently. This is scoped to one construction, set and cleared in a
     * {@code finally}, and is not readable afterwards — {@link #storageOptions()} is.
     *
     * <p>It defaults to the properties, so nothing that does not use {@link #withStorageOptions}
     * changes behaviour.
     */
    private static final ThreadLocal<org.beehive.jllm.runtime.policy.StorageOptions>
            STORAGE_FOR_CONSTRUCTION = new ThreadLocal<>();

    /**
     * What the state under construction should allocate: what was handed in, or the properties.
     *
     * <p>Deliberately <b>not</b> {@code ThreadLocal.withInitial(.)}: that caches the first value
     * per thread, so a property set later in the same JVM would never be seen — which is the very
     * defect this replaces, reintroduced one layer down. It cost a red test to notice.
     */
    private static org.beehive.jllm.runtime.policy.StorageOptions storageForConstruction() {
        var handedIn = STORAGE_FOR_CONSTRUCTION.get();
        return handedIn != null
                ? handedIn
                : org.beehive.jllm.runtime.policy.StorageOptions.fromSystemProperties();
    }

    /**
     * Builds a state with these storage options, and restores the previous default afterwards.
     *
     * @param storage what the state should allocate
     * @param build the construction, typically a {@code Model::createNewState} call
     */
    public static <T> T withStorageOptions(
            org.beehive.jllm.runtime.policy.StorageOptions storage,
            java.util.function.Supplier<T> build) {
        var previous = STORAGE_FOR_CONSTRUCTION.get();
        STORAGE_FOR_CONSTRUCTION.set(java.util.Objects.requireNonNull(storage, "storage"));
        try {
            return build.get();
        } finally {
            if (previous == null) {
                STORAGE_FOR_CONSTRUCTION.remove();
            } else {
                STORAGE_FOR_CONSTRUCTION.set(previous);
            }
        }
    }

    private static final ThreadLocal<Integer> PREFILL_BATCH_FOR_CONSTRUCTION = new ThreadLocal<>();

    /**
     * The batch width the state under construction should size its prefill workspace for.
     *
     * <p>The width has to be known at allocation, not at plan construction: the batch arrays are
     * sized from it, and a plan built for a wider batch than was allocated binds a null buffer and
     * fails inside TornadoVM with {@code null object passed into streamIn()} rather than anything
     * that names the cause. The CLI supplies it as a system property; a facade caller supplies it
     * as an {@code ExecutionPolicy}, which is resolved after construction, so it is scoped around
     * the construction instead.
     */
    private static int prefillBatchForConstruction() {
        Integer handedIn = PREFILL_BATCH_FOR_CONSTRUCTION.get();
        return handedIn != null ? handedIn : Integer.getInteger("jllm.prefillBatchSize", 1);
    }

    /**
     * Builds a state sized for this prefill batch width, restoring the previous default after.
     *
     * @param prefillBatchSize the width the prefill workspace must accommodate
     * @param build the construction, typically a {@code Model::createNewState} call
     */
    public static <T> T withPrefillBatchSize(
            int prefillBatchSize, java.util.function.Supplier<T> build) {
        Integer previous = PREFILL_BATCH_FOR_CONSTRUCTION.get();
        PREFILL_BATCH_FOR_CONSTRUCTION.set(prefillBatchSize);
        try {
            return build.get();
        } finally {
            if (previous == null) {
                PREFILL_BATCH_FOR_CONSTRUCTION.remove();
            } else {
                PREFILL_BATCH_FOR_CONSTRUCTION.set(previous);
            }
        }
    }

    protected State(Configuration config, int batchsize) {
        this(config, batchsize, null);
    }

    /**
     * How this session executes, resolved once and read at plan construction.
     *
     * <p>It lives here because the state is the one per-session object every plan and every layer
     * already holds — putting it here is what let the readers migrate off class constants without
     * threading a parameter through forty construction signatures.
     *
     * <p>Defaulted from the system properties so a state built outside the facade — the CLI, the
     * benchmark harness, a test — behaves exactly as it did when these were {@code static final}
     * fields. A session replaces it once, before anything reads it.
     */
    private org.beehive.jllm.runtime.policy.ExecutionPolicy executionPolicy =
            org.beehive.jllm.runtime.policy.ExecutionPolicy.fromSystemProperties();

    /** Set once a plan has read the policy, so a later change is refused rather than ignored. */
    private boolean executionPolicyRead;

    /**
     * This state's execution policy. Read at <b>plan construction</b>, never per token.
     *
     * <p>Reading it locks it: a policy changed after a plan was built would describe a program
     * nobody compiled, and silently doing nothing is exactly the failure this migration removes.
     */
    public org.beehive.jllm.runtime.policy.ExecutionPolicy executionPolicy() {
        executionPolicyRead = true;
        return executionPolicy;
    }

    /**
     * Resolves this state's policy, once, before any plan is built.
     *
     * @throws IllegalStateException if a plan has already read the policy
     */
    public void resolveExecutionPolicy(org.beehive.jllm.runtime.policy.ExecutionPolicy policy) {
        if (executionPolicyRead) {
            throw new IllegalStateException(
                    "the execution policy was already read by a plan;"
                            + " changing it now would describe a program that was never compiled");
        }
        this.executionPolicy = java.util.Objects.requireNonNull(policy, "policy");
    }

    protected State(
            Configuration config, int batchsize, org.beehive.jllm.runtime.kv.KvLease lease) {
        // Assigned before createStateFields, which the subclass overrides and which needs to know
        // whether there is leased storage to bind rather than arrays to allocate.
        this.storageOptions = storageForConstruction();
        // The width the prefill workspace is sized from, taken once. A family that allocates its
        // own chunk-wide buffers reads this rather than the execution policy: the policy is
        // resolved per generation and says nothing about how this state was built.
        this.prefillBatchWidth = prefillBatchForConstruction();
        this.kvLease = lease;
        this.kvSlot = lease != null ? lease.slot() : 0;
        this.batchsize = batchsize;
        this.latestToken = -1;
        this.localSize = 256;

        // Initialize all fields through the creation method
        // The workspace exists before the family fills it: a family says how large, and the
        // backend's allocator says what with.
        this.workspace = new org.beehive.jllm.backend.tornado.workspace.TornadoWorkspace();
        StateFields fields = createStateFields(config);

        this.x = fields.x;
        this.xb = fields.xb;
        this.xb2 = fields.xb2;
        this.hb = fields.hb;
        this.hb2 = fields.hb2;
        this.q = fields.q;
        this.k = fields.k;
        this.v = fields.v;
        this.att = fields.att;
        this.logits = fields.logits;
        // int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        this.keyCache = fields.keyCache;
        this.valueCache = fields.valueCache;

        // The workspace exists before the family fills it: a family says how large, and the
        // backend's allocator says what with.

        // dim vs kvdim
        // The slot is data the kernels read, so it is written once here and re-written by
        // resetPositionHolder(); a bare init(0) would silently address slot 0.
        TornadoWorkspaces.writeSlot(workspace, this.kvSlot);
        this.kvBlockCfg = fields.kvBlockCfg;
        this.kvBlockStride = fields.kvBlockStride;

        // You need at least 9 elements: 1 for the final result + 8 for the workgroup partial sums

        int gpuBatchSize = prefillBatchWidth;
        if (gpuBatchSize > 1) {
            // The tensor-core GEMM kernels operate on full 128-row M tiles
            // (BM = 128). Pad the GEMM-adjacent activation buffers so any
            // batch size launches whole tiles; rows >= gpuBatchSize hold
            // garbage and are never read by the non-GEMM kernels.
            int paddedGpuBatch = (gpuBatchSize + 127) & ~127;
            int qDim = batchQDim(config);
            int hiddenDimBatch = batchHiddenDim(config);
            int kvDim = batchKvDim(config);
            this.workspace.embeddingXBatch =
                    TornadoWorkspaces.halfFloats(gpuBatchSize * config.dim());
            this.workspace.wrapXBatch = TornadoWorkspaces.floats(gpuBatchSize * config.dim());
            this.workspace.wrapXbFP16Batch =
                    TornadoWorkspaces.halfFloats(paddedGpuBatch * config.dim());
            this.workspace.wrapQBatch = TornadoWorkspaces.floats(gpuBatchSize * qDim);
            this.workspace.wrapKBatch = TornadoWorkspaces.floats(gpuBatchSize * kvDim);
            this.workspace.wrapVBatch = TornadoWorkspaces.floats(gpuBatchSize * kvDim);
            this.workspace.wrapXbBatch = TornadoWorkspaces.floats(gpuBatchSize * qDim);
            this.workspace.wrapHbBatch = TornadoWorkspaces.floats(gpuBatchSize * hiddenDimBatch);
            this.workspace.attnScaleBatch = TornadoWorkspaces.floats(gpuBatchSize);
            this.workspace.ffnScaleBatch = TornadoWorkspaces.floats(gpuBatchSize);
            this.workspace.batchStartPosHolder = TornadoWorkspaces.ints(3);
            this.workspace.normedXFFNFP16 =
                    TornadoWorkspaces.halfFloats(paddedGpuBatch * config.dim());
            this.workspace.ffnGateResult = TornadoWorkspaces.floats(gpuBatchSize * hiddenDimBatch);
            this.workspace.ffnUpResult = TornadoWorkspaces.floats(gpuBatchSize * hiddenDimBatch);

            this.workspace.xbFP16Batch = TornadoWorkspaces.halfFloats(gpuBatchSize * config.dim());
            this.workspace.attnOutFP16 =
                    TornadoWorkspaces.halfFloats(
                            paddedGpuBatch
                                    * qDim); // qDim == dim for Llama; qDim = nHeads*headDim for
            // Qwen3
            this.workspace.woOut = TornadoWorkspaces.floats(paddedGpuBatch * config.dim());
            this.workspace.wrapHbFP16Batch =
                    TornadoWorkspaces.halfFloats(paddedGpuBatch * hiddenDimBatch);
            this.workspace.w2Out = TornadoWorkspaces.floats(paddedGpuBatch * config.dim());
            this.workspace.qkvResultBatch =
                    TornadoWorkspaces.floats(paddedGpuBatch * (qDim + 2 * kvDim));
            this.workspace.gateUpResultBatch =
                    TornadoWorkspaces.floats(paddedGpuBatch * 2 * hiddenDimBatch);
        } else {
            this.workspace.embeddingXBatch = null;
            this.workspace.wrapXBatch = null;
            this.workspace.wrapXbFP16Batch = null;
            this.workspace.wrapQBatch = null;
            this.workspace.wrapKBatch = null;
            this.workspace.wrapVBatch = null;
            this.workspace.wrapXbBatch = null;
            this.workspace.wrapHbBatch = null;
            this.workspace.attnScaleBatch = null;
            this.workspace.ffnScaleBatch = null;
            this.workspace.batchStartPosHolder = null;
            this.workspace.normedXFFNFP16 = null;
            this.workspace.ffnGateResult = null;
            this.workspace.ffnUpResult = null;
            this.workspace.xbFP16Batch = null;
            this.workspace.attnOutFP16 = null;
            this.workspace.woOut = null;
            this.workspace.wrapHbFP16Batch = null;
            this.workspace.w2Out = null;
            this.workspace.qkvResultBatch = null;
            this.workspace.gateUpResultBatch = null;
        }
    }

    /**
     * Q-projection output dimension per token (model specific: = dim for Llama; differs for Qwen3).
     */
    protected int batchQDim(Configuration config) {
        return config.dim();
    }

    /**
     * Feed-forward width the chunk-wide buffers are sized from.
     *
     * <p>A hook rather than a direct {@code config.hiddenDim()} for the same reason {@link
     * #batchQDim} is one: a family whose feed-forward width differs by layer has no single hidden
     * dimension to report, and {@code hiddenDim()} on such a configuration refuses to answer rather
     * than pick a layer. Such a family returns its widest here, which is what one buffer shared by
     * every layer's graph has to hold.
     */
    protected int batchHiddenDim(Configuration config) {
        return config.hiddenDim();
    }

    /**
     * Fills in a state's KV wrappers: from the lease's shared storage when there is one, in the
     * block-major layout when paged, contiguously otherwise.
     *
     * <p>Every family that has migrated calls this instead of allocating its own key and value
     * arrays, so the three shapes are decided once. The families that have not migrated do not call
     * it at all and keep allocating exactly what they always did.
     *
     * @param fields the state under construction
     * @param config the model configuration
     * @param kvDim KV values per token — {@code nEmbdGqa} for Qwen, {@code kvDim} for Llama
     * @param useFp16 allocate the half-precision pair as well (ignored when leased: the store
     *     already chose one precision)
     * @return whether the KV came from leased storage rather than being allocated here
     */
    /**
     * Fills in a state's KV wrappers: from the lease's shared storage when there is one, and
     * block-major in arrays of its own when there is not.
     *
     * @param fields the state under construction
     * @param config the model configuration
     * @param kvDim KV values per token — {@code nEmbdGqa} for Qwen, {@code kvDim} for Llama
     * @param useFp16 allocate the half-precision pair as well (ignored when leased: the store
     *     already chose one precision)
     * @return whether the KV came from leased storage rather than being allocated here
     */
    protected boolean fillKvFields(
            StateFields fields, Configuration config, int kvDim, boolean useFp16) {
        return fillKvFields(fields, config, kvDim, config.numberOfLayers(), useFp16);
    }

    /**
     * {@link #fillKvFields(StateFields, Configuration, int, boolean)} for a family where not every
     * layer holds key/value entries.
     *
     * <p>{@code qwen35} attends in one layer of four; the other three mix with a recurrence and
     * have nothing to retain. Sizing the store by the layer count would allocate four times what
     * the model uses — gigabytes at any useful context — so the store is sized by the layers that
     * actually write to it, and those layers address it by a <b>dense</b> index rather than their
     * own. The kernels need no change: {@code layer} is already just a stride multiplier to them.
     *
     * @param kvLayers how many layers hold key/value entries; the dense index space they address
     */
    protected boolean fillKvFields(
            StateFields fields, Configuration config, int kvDim, int kvLayers, boolean useFp16) {
        // The caller says whether this family has FP16 kernels at all; the storage options say
        // whether they were asked for. Both must hold.
        useFp16 = useFp16 && storageOptions.usesFp16KeyValueCache();
        org.beehive.jllm.runtime.kv.KvStorage storage = kvLease != null ? kvLease.storage() : null;
        if (storage != null) {
            // Leased: the backend writes its own arrays in, and this state never learns what they
            // are. Nothing KV-shaped is allocated here — that is the whole point.
            //
            // No lease, or a lease without storage, falls through to private allocation below: that
            // means no backend storage was requested, and a CPU-only state is valid. Storage that
            // no binder claims is a different situation and throws, because falling back would give
            // correct output, more memory and no explanation on a machine that asked for a shared
            // pool. The likeliest cause is a shaded jar that lost the service file.
            int[] layout = new int[2];
            org.beehive.jllm.backend.tornado.workspace.TornadoWorkspaces.bindLeasedKeyValue(
                    workspace, kvLease, layout);
            // The layout is the store's, not recomputed here: a state that derived its own would
            // address a pool laid out differently.
            fields.kvBlockCfg = layout[0];
            fields.kvBlockStride = layout[1];
            return true;
        }

        // Its own arrays, laid out [block][layer][posInBlock][c] — a permutation of the contiguous
        // layout plus at most blockSize-1 positions of padding.
        int blocksPerSeq = (config.contextLength() + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
        int kvElements = blocksPerSeq * kvLayers * KV_BLOCK_SIZE * kvDim;
        fields.kvBlockCfg = KV_BLOCK_SIZE | (blocksPerSeq << 16);
        fields.kvBlockStride = kvLayers * KV_BLOCK_SIZE * kvDim;
        // One sequence's private table, identity-mapped: logical block i is physical block i.
        // Addressed only at slot 0 — it is sized blocksPerSeq, so no larger slot fits in it.
        TornadoWorkspaces.identityBlockTable(workspace, blocksPerSeq);
        TornadoWorkspaces.privateKeyValueFP32(workspace, kvElements);
        if (useFp16) {
            workspace.wrapKeyCacheFP16 = TornadoWorkspaces.zeroedHalfFloats(kvElements);
            workspace.wrapValueCacheFP16 = TornadoWorkspaces.zeroedHalfFloats(kvElements);
        }
        return false;
    }

    /**
     * Whether this state's key/value cache is held in half precision.
     *
     * <p>Answered from what was <b>allocated</b>, not from the property that requested it. Every
     * reader used to write {@code State.USE_FP16_KV && state.workspace.wrapKeyCacheFP16 != null} —
     * the property <i>and</i> the null check, because the property alone is not the truth: a family
     * whose state has no FP16 arrays, or a leased state whose pool was built in FP32, holds FP32
     * whatever the property says. Asking the state removes both the global read and the chance of
     * writing only half of that condition.
     */
    public boolean usesFp16KeyValueCache() {
        return workspace.wrapKeyCacheFP16 != null;
    }

    /**
     * Sets the decode position, keeping the KV slot alongside it.
     *
     * <p>Use this rather than writing {@code positionHolder} directly: the slot lives in the same
     * buffer, and a write that forgets it points the kernels at slot 0 — another session's KV once
     * storage is shared.
     */
    public void setPosition(int position) {
        TornadoWorkspaces.setPosition(workspace, position, kvSlot);
    }

    /** Clears the position holder for a fresh plan execution, preserving the slot. */
    public void resetPositionHolder() {
        TornadoWorkspaces.resetPosition(workspace, kvSlot);
    }

    /**
     * KV-cache dimension per token (model specific: = dim*nHeadKv/nHeads for Llama; differs for
     * Qwen3).
     */
    protected int batchKvDim(Configuration config) {
        return (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
    }

    /**
     * Discards whatever this state carries that a position rewind does not.
     *
     * <p>A no-op for every family whose only per-sequence memory is a key/value cache: attention
     * reads no further than the current position, so rewinding the position <i>is</i> the reset,
     * and clearing the cache would cost a context-length write for no effect.
     *
     * <p>It is not a no-op for a family with recurrent state. A convolution window or a delta-net
     * matrix has summed the whole sequence into a fixed-size buffer with no notion of position, so
     * a reused session would condition the new sequence on the old one — fluent output, silently
     * wrong. Such a family overrides this.
     */
    public void resetSequenceState() {}

    /**
     * The device buffers that hold this family's recurrent state, if it keeps any.
     *
     * <p>{@link #resetSequenceState()} zeroes the host side, and for these buffers that is not
     * enough: they are declared {@code FIRST_EXECUTION} and then {@code persistOnDevice}, so the
     * device copy is written by the kernels and never re-read from the host. A reset that only
     * clears the host arrays leaves the accelerator continuing the previous sequence. The plan
     * uploads these after a reset; a family with no recurrent state returns nothing.
     */
    public Object[] recurrentDeviceBuffers() {
        return new Object[0];
    }

    // Abstract method - subclasses implement their specific allocation logic and sizes
    protected abstract StateFields createStateFields(Configuration config);

    /**
     * A host key/value tensor in the representation this state's storage selects: half precision
     * for an FP16 cache, single precision otherwise. The CPU forward pass reads and writes it
     * through {@code FloatTensor}, so accumulation stays FP32 either way.
     */
    protected final org.beehive.jllm.tensor.standard.FloatTensor allocateKeyValue(int... dims) {
        return storageOptions.usesFp16KeyValueCache()
                ? org.beehive.jllm.tensor.standard.ArrayHalfFloatTensor.allocate(dims)
                : org.beehive.jllm.tensor.standard.ArrayFloatTensor.allocate(dims);
    }

    /** The host tensors a family allocates during construction. */
    static class StateFields {
        public FloatTensor x, xb, xb2, hb, hb2, q, k, v, att, logits;
        public FloatTensor[] keyCache, valueCache;
        public int kvBlockCfg;
        public int kvBlockStride;
    }

    @Override
    public State clone() throws CloneNotSupportedException {
        return (State) super.clone();
    }
}
