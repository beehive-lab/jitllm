package org.beehive.jitllm.inference.state;

import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.deepseek2.DeepSeek2Configuration;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * A {@code deepseek2} session.
 *
 * <p>The key/value store holds one {@link DeepSeek2Configuration#keyWidth()}-wide row per position
 * and layer — the normalized latent followed by the rotated key — in {@link #keyCache}. The value
 * is the latent half of the same row, so {@link #valueCache} holds nothing.
 */
public final class DeepSeek2State extends State {

    /** The compressed query, before its norm and {@code attn_q_b}. */
    public final FloatTensor queryLatent;

    /** Every head's absorbed query: the latent-space no-rope part, then the rotated part. */
    public final FloatTensor absorbedQuery;

    /** Every head's attended latent, before {@code attn_v_b}. */
    public final FloatTensor latentOut;

    /** {@code attn_kv_a_mqa}'s output: the latent, then the rotated key. */
    public final FloatTensor compressedKeyValue;

    public final FloatTensor routerScores;
    public final FloatTensor selectionScores;
    public final int[] expertIds;
    public final float[] expertWeights;
    public final FloatTensor expertHidden;
    public final FloatTensor expertHiddenUp;
    public final FloatTensor expertOut;

    public DeepSeek2State(Configuration config, int batchsize) {
        this(config, batchsize, null);
    }

    public DeepSeek2State(
            Configuration config, int batchsize, org.beehive.jitllm.runtime.kv.KvLease lease) {
        super(config, batchsize, lease);
        DeepSeek2Configuration c = (DeepSeek2Configuration) config;
        this.queryLatent = ArrayFloatTensor.allocate(Math.max(1, c.qLoraRank()));
        this.absorbedQuery = ArrayFloatTensor.allocate(c.absorbedQueryDim());
        this.latentOut = ArrayFloatTensor.allocate(c.latentOutputDim());
        this.compressedKeyValue = ArrayFloatTensor.allocate(c.compressedKeyValueDim());
        this.routerScores = ArrayFloatTensor.allocate(c.expertCount());
        this.selectionScores = ArrayFloatTensor.allocate(c.expertCount());
        this.expertIds = new int[c.expertsUsed()];
        this.expertWeights = new float[c.expertsUsed()];
        int width = Math.max(c.expertHiddenDim(), c.sharedHiddenDim());
        this.expertHidden = ArrayFloatTensor.allocate(width);
        this.expertHiddenUp = ArrayFloatTensor.allocate(width);
        this.expertOut = ArrayFloatTensor.allocate(c.dim());
        if (Boolean.parseBoolean(System.getProperty("use.tornadovm", "false"))
                && prefillBatchWidth > 1) {
            allocateBatchWorkspace(c);
        }
    }

    // ── Batched prefill ───────────────────────────────────────────────────────

    /** {@code attn_kv_a_mqa}'s output padded to whole 128-column GEMM tiles. */
    public static final int KV_A_PADDED = 640;

    /** The chunk width rounded up to whole 128-row GEMM tiles. */
    public int paddedBatch;

    /** The context rounded up to whole 128-column GEMM tiles: the score GEMM's width. */
    public int paddedPositions;

    // @formatter:off
    /**
     * The chunk's buffers: the projections' outputs, the attention's (token, head) rows, the int8
     * GEMMs' quantized activation, and the routed and shared experts' activations.
     */
    // @formatter:on
    private void allocateBatchWorkspace(DeepSeek2Configuration c) {
        int p = (prefillBatchWidth + 127) & ~127;
        this.paddedBatch = p;
        this.paddedPositions = (c.contextLength() + 127) & ~127;
        int heads = c.numberOfHeads();
        int keyWidth = c.keyWidth();
        int rank = c.kvLoraRank();
        int dim = c.dim();
        int rows = p * heads;

        workspace.wrapNormedBatch = TornadoWorkspaces.floats(p * dim);
        workspace.queryLatentBatch = TornadoWorkspaces.floats(p * c.qLoraRank());
        workspace.compressedKvBatch = TornadoWorkspaces.floats(p * KV_A_PADDED);
        workspace.queryBatch = TornadoWorkspaces.floats(p * c.queryDim());
        workspace.queryF16Batch = TornadoWorkspaces.halfFloats(p * c.queryDim());
        workspace.absorbedBatch = TornadoWorkspaces.floats(rows * keyWidth);
        workspace.absorbedF16Batch = TornadoWorkspaces.halfFloats(rows * keyWidth);
        workspace.keysF16 = TornadoWorkspaces.halfFloats(paddedPositions * keyWidth);
        workspace.latentTF16 = TornadoWorkspaces.halfFloats(rank * paddedPositions);
        workspace.scoresBatch = TornadoWorkspaces.floats(rows * paddedPositions);
        workspace.probsF16Batch = TornadoWorkspaces.halfFloats(rows * paddedPositions);
        workspace.latentBatch = TornadoWorkspaces.floats(rows * rank);
        workspace.latentF16Batch = TornadoWorkspaces.halfFloats(rows * rank);
        workspace.attnOutBatch = TornadoWorkspaces.floats(p * c.attentionOutputInputDim());
        workspace.gateBatch = TornadoWorkspaces.floats(p * c.hiddenDim());
        workspace.hiddenBatch = TornadoWorkspaces.floats(p * c.hiddenDim());

        int widest =
                Math.max(
                        Math.max(dim, c.hiddenDim()),
                        Math.max(c.attentionOutputInputDim(), c.qLoraRank()));
        workspace.wrapQ8ActBatch = TornadoWorkspaces.bytes(p * widest);
        workspace.wrapQ8ActScales = TornadoWorkspaces.floats(p * widest / 32);
        workspace.wrapQ8SplitPartial = TornadoWorkspaces.floats(8 * p * dim);

        int experts = Math.max(1, c.expertCount());
        int assignments = p * Math.max(1, c.expertsUsed());
        int hidden = Math.max(32, c.expertHiddenDim());
        int shared = Math.max(32, c.sharedHiddenDim());
        workspace.wrapMoeLogitsBatch = TornadoWorkspaces.floats(p * experts);
        workspace.wrapMoeIdsBatch = TornadoWorkspaces.ints(assignments);
        workspace.wrapMoeWeightsBatch = TornadoWorkspaces.floats(assignments);
        workspace.wrapMoeSharedGateBatch = TornadoWorkspaces.floats(p);
        workspace.wrapMoeSortedToken = TornadoWorkspaces.ints(assignments);
        workspace.wrapMoePosition = TornadoWorkspaces.ints(assignments);
        workspace.wrapMoeTiles =
                TornadoWorkspaces.ints(
                        1
                                + 3
                                        * org.beehive.jitllm.backend.tornado.kernels
                                                .Qwen35MoeBatchKernels.maxTiles(
                                                assignments, experts));
        workspace.wrapMoeHiddenBatch = TornadoWorkspaces.floats(assignments * hidden);
        workspace.wrapMoeHiddenQ8 = TornadoWorkspaces.bytes(assignments * hidden);
        workspace.wrapMoeHiddenScales = TornadoWorkspaces.floats(assignments * hidden / 32);
        workspace.wrapMoeOutBatch = TornadoWorkspaces.floats(assignments * dim);
        workspace.wrapMoeSharedGateUp = TornadoWorkspaces.floats(p * shared);
        workspace.wrapMoeSharedHidden = TornadoWorkspaces.floats(p * shared);
        workspace.wrapMoeSharedQ8 = TornadoWorkspaces.bytes(p * shared);
        workspace.wrapMoeSharedScales = TornadoWorkspaces.floats(p * shared / 32);
        workspace.wrapMoeSharedOut = TornadoWorkspaces.floats(p * dim);
    }

    @Override
    protected int batchQDim(Configuration config) {
        return ((DeepSeek2Configuration) config).absorbedQueryDim();
    }

    @Override
    protected int batchKvDim(Configuration config) {
        return ((DeepSeek2Configuration) config).keyWidth();
    }

    @Override
    protected StateFields createStateFields(Configuration configuration) {
        DeepSeek2Configuration config = (DeepSeek2Configuration) configuration;
        StateFields fields = new StateFields();
        int dim = config.dim();
        int hidden = Math.max(config.hiddenDim(), config.expertHiddenDim());
        fields.x = ArrayFloatTensor.allocate(dim);
        fields.xb = ArrayFloatTensor.allocate(Math.max(dim, config.attentionOutputInputDim()));
        fields.xb2 = ArrayFloatTensor.allocate(dim);
        fields.hb = ArrayFloatTensor.allocate(hidden);
        fields.hb2 = ArrayFloatTensor.allocate(hidden);
        fields.q = ArrayFloatTensor.allocate(config.queryDim());
        fields.k = ArrayFloatTensor.allocate(config.keyWidth());
        fields.v = ArrayFloatTensor.allocate(config.kvLoraRank());
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());
        int layers = config.numberOfLayers();
        fields.keyCache = new FloatTensor[layers];
        fields.valueCache = new FloatTensor[layers];
        boolean device = Boolean.parseBoolean(System.getProperty("use.tornadovm", "false"));
        for (int l = 0; l < layers; l++) {
            // The host cache only where the host runs; a device session keeps its own below.
            fields.keyCache[l] =
                    allocateKeyValue(device ? 1 : config.contextLength(), config.keyWidth());
        }
        if (device) {
            allocateDeviceWorkspace(config);
        }
        return fields;
    }

    // ── Cache layout ──────────────────────────────────────────────────────────

    /**
     * The first layer this state's device cache holds: 0, unless the state is one stage of a model
     * split across devices, whose provider sets it to the stage's first layer.
     */
    public int cacheFirstLayer;

    /**
     * Where layer {@code l}'s rows start in the device cache: {@code [layer][position][keyWidth]}.
     */
    public int cacheLayerOffset(int l) {
        return (l - cacheFirstLayer) * cacheLayerStride;
    }

    private int cacheLayerStride;

    // @formatter:off
    /**
     * The device arrays of the single-token plan.
     *
     * <p>The cache is one row of {@link DeepSeek2Configuration#keyWidth()} per position and layer —
     * the normalized latent and the rotated key — and is the only key/value store: the value is the
     * row's latent half. {@code wrapValueCache} is a placeholder the shared graphs can name.
     */
    // @formatter:on
    private void allocateDeviceWorkspace(DeepSeek2Configuration config) {
        // A stage of a split holds its own layers only (keyValueLayersForConstruction).
        int layers = keyValueLayersForConstruction(config.numberOfLayers());
        this.cacheLayerStride = config.contextLength() * config.keyWidth();
        int dim = config.dim();

        TornadoWorkspaces.activationQ8_0(workspace, dim);
        workspace.wrapX = TornadoWorkspaces.floats(dim);
        workspace.wrapXb =
                TornadoWorkspaces.floats(Math.max(dim, config.attentionOutputInputDim()));
        workspace.wrapXb2 = TornadoWorkspaces.floats(dim);
        int widest =
                Math.max(
                        Math.max(dim, config.hiddenDim()),
                        Math.max(config.attentionOutputInputDim(), config.qLoraRank()));
        TornadoWorkspaces.packedActivation(workspace, widest);
        workspace.wrapHb = TornadoWorkspaces.floats(config.hiddenDim());
        workspace.wrapLogits = TornadoWorkspaces.floats(config.vocabularySize());

        workspace.wrapQueryLatent = TornadoWorkspaces.floats(config.qLoraRank());
        workspace.wrapQ = TornadoWorkspaces.floats(config.queryDim());
        workspace.wrapCompressedKv = TornadoWorkspaces.floats(config.compressedKeyValueDim());
        workspace.wrapAbsorbedQuery = TornadoWorkspaces.floats(config.absorbedQueryDim());
        workspace.wrapLatentOut = TornadoWorkspaces.floats(config.latentOutputDim());
        workspace.wrapAtt =
                TornadoWorkspaces.floats(config.numberOfHeads() * config.contextLength());
        workspace.wrapAttSplit =
                TornadoWorkspaces.floats(
                        config.numberOfHeads()
                                * DeepSeek2Configuration.DECODE_ATTENTION_SPLITS
                                * (config.kvLoraRank() + 2));
        int cacheElements = Math.max(1, layers * cacheLayerStride);
        if (storageOptions().usesFp16KeyValueCache()) {
            workspace.wrapKeyCacheFP16 = TornadoWorkspaces.halfFloats(cacheElements);
            workspace.wrapValueCacheFP16 = TornadoWorkspaces.halfFloats(1);
        } else {
            workspace.wrapKeyCache = TornadoWorkspaces.floats(cacheElements);
            workspace.wrapValueCache = TornadoWorkspaces.floats(1);
        }

        int experts = config.expertCount();
        int used = config.expertsUsed();
        workspace.wrapRouterLogits = TornadoWorkspaces.floats(Math.max(1, experts));
        workspace.wrapSelectedExperts = TornadoWorkspaces.ints(Math.max(1, used));
        workspace.wrapRoutingWeights = TornadoWorkspaces.floats(Math.max(1, used));
        workspace.wrapSharedGate = TornadoWorkspaces.floats(1);
        int moeHidden = Math.max(32, config.routedHiddenDim() + config.sharedHiddenDim());
        workspace.wrapMoeHidden = TornadoWorkspaces.floats(moeHidden);
        workspace.wrapMoeHiddenQuants = TornadoWorkspaces.ints(moeHidden / 4);
        workspace.wrapMoeHiddenQScales = TornadoWorkspaces.floats(moeHidden / 32);
        workspace.wrapMoeHiddenQSums = TornadoWorkspaces.ints(moeHidden / 32);

        // [0] = position, [1] = table-local KV slot.
        workspace.positionHolder = TornadoWorkspaces.ints(2);
        workspace.temp = TornadoWorkspaces.floats(1 + ((dim + localSize - 1) / localSize));
        workspace.tempFFN = TornadoWorkspaces.floats(1 + ((dim + localSize - 1) / localSize));
        workspace.tempLogits = TornadoWorkspaces.floats(1 + ((dim + localSize - 1) / localSize));
    }
}
