package org.beehive.jitllm.inference.state;

import java.util.Arrays;
import org.beehive.jitllm.backend.tornado.workspace.DeepSeek4BatchArrays;
import org.beehive.jitllm.backend.tornado.workspace.DeepSeek4DeviceArrays;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.deepseek4.DeepSeek4Configuration;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * A {@code deepseek4} session on the host.
 *
 * <p>Per layer (the MTP layer included):
 *
 * <ul>
 *   <li>the attention window, a ring of {@link DeepSeek4Configuration#slidingWindow()} key/value
 *       vectors, slot {@code position % window}; it is {@link #keyCache};
 *   <li>with a compression ratio {@code r}, the compressor's pending entries — for the overlapping
 *       ratio 4, the previous block's then the current block's, {@code 2r} rows of {@code 2 *
 *       headDim} projections and gate scores; for ratio 128, {@code r} rows of {@code headDim} —
 *       and the compressed key of every complete block;
 *   <li>with an indexer, its own compressor's entries and compressed keys, {@code indexerHeadDim}
 *       wide.
 * </ul>
 *
 * <p>The pending entries are the only state a new sequence must clear ({@link
 * #resetSequenceState()}): the previous block of the first block has scores of minus infinity, so
 * it carries no weight. A query reads compressed keys only up to its own position.
 */
public final class DeepSeek4State extends State {

    /** The residual streams, stream-major: {@code streams[n * dim + i]}. */
    public final FloatTensor streams;

    /** The streams as they entered the current sublayer, mixed back in after it. */
    public final FloatTensor residualStreams;

    public final float[] hcMixes;
    public final float[] hcPre;
    public final float[] hcPost;

    /** {@code hcComb[m * n + k]}: how much input stream {@code m} feeds output stream {@code k}. */
    public final float[] hcComb;

    /** The RoPE frequencies of the layers without compression, and of the compressed layers. */
    public final float[] ropeFrequencies;

    public final float[] compressRopeFrequencies;

    // Attention.
    public final FloatTensor queryLatent;
    public final FloatTensor keyValue;
    public final FloatTensor attentionOut;
    public final FloatTensor groupedOut;

    /** The window positions, then the compressed blocks, the current query attends to. */
    public final int[] attendedPositions;

    public final int[] attendedBlocks;
    public int attendedPositionCount;
    public int attendedBlockCount;

    // Compressors, per layer (null without one).
    public final FloatTensor[] compressorKv;
    public final FloatTensor[] compressorScore;
    public final FloatTensor[] compressedKeys;
    public final FloatTensor compressorProjection;
    public final FloatTensor compressorGate;
    public final FloatTensor pooled;

    // Indexers, per layer (null without one).
    public final FloatTensor[] indexerKv;
    public final FloatTensor[] indexerScore;
    public final FloatTensor[] indexerKeys;
    public final FloatTensor indexQuery;
    public final FloatTensor indexHeadWeights;
    public final float[] blockScores;

    // Feed-forward.
    public final FloatTensor routerScores;
    public final FloatTensor selectionScores;
    public final int[] expertIds;
    public final float[] expertWeights;
    public final FloatTensor expertHidden;
    public final FloatTensor expertHiddenUp;
    public final FloatTensor expertOut;

    public DeepSeek4State(Configuration config, int batchsize) {
        this(config, batchsize, null);
    }

    public DeepSeek4State(
            Configuration config, int batchsize, org.beehive.jitllm.runtime.kv.KvLease lease) {
        super(config, batchsize, lease);
        DeepSeek4Configuration c = (DeepSeek4Configuration) config;
        int dim = c.dim();
        int n = c.hcCount();
        int layers = c.allLayers();
        int hd = c.headDim();
        int id = c.indexerHeadDim();

        this.streams = ArrayFloatTensor.allocate(n * dim);
        this.residualStreams = ArrayFloatTensor.allocate(n * dim);
        this.hcMixes = new float[c.hcMixCount()];
        this.hcPre = new float[n];
        this.hcPost = new float[n];
        this.hcComb = new float[n * n];
        this.ropeFrequencies = c.ropeFrequencies(false);
        this.compressRopeFrequencies = c.ropeFrequencies(true);

        this.queryLatent = ArrayFloatTensor.allocate(c.qLoraRank());
        this.keyValue = ArrayFloatTensor.allocate(hd);
        this.attentionOut = ArrayFloatTensor.allocate(c.queryDim());
        this.groupedOut = ArrayFloatTensor.allocate(c.groupedOutputDim());
        this.attendedPositions = new int[c.slidingWindow()];
        this.attendedBlocks = new int[Math.max(1, c.maxAttended())];

        this.compressorKv = new FloatTensor[layers];
        this.compressorScore = new FloatTensor[layers];
        this.compressedKeys = new FloatTensor[layers];
        this.indexerKv = new FloatTensor[layers];
        this.indexerScore = new FloatTensor[layers];
        this.indexerKeys = new FloatTensor[layers];
        int indexedBlocks = 1;
        for (int l = 0; l < layers; l++) {
            int r = c.ratio(l);
            if (r == 0) {
                continue;
            }
            int o = DeepSeek4Configuration.overlap(r);
            compressorKv[l] = ArrayFloatTensor.allocate(o * r * o * hd);
            compressorScore[l] = ArrayFloatTensor.allocate(o * r * o * hd);
            compressedKeys[l] = ArrayFloatTensor.allocate(c.maxBlocks(r) * hd);
            if (c.hasIndexer(l)) {
                indexerKv[l] = ArrayFloatTensor.allocate(2 * r * 2 * id);
                indexerScore[l] = ArrayFloatTensor.allocate(2 * r * 2 * id);
                indexerKeys[l] = ArrayFloatTensor.allocate(c.maxBlocks(r) * id);
                indexedBlocks = Math.max(indexedBlocks, c.maxBlocks(r));
            }
        }
        this.compressorProjection = ArrayFloatTensor.allocate(2 * Math.max(hd, id));
        this.compressorGate = ArrayFloatTensor.allocate(2 * Math.max(hd, id));
        this.pooled = ArrayFloatTensor.allocate(Math.max(hd, id));
        this.indexQuery = ArrayFloatTensor.allocate(Math.max(1, c.indexerHeads() * id));
        this.indexHeadWeights = ArrayFloatTensor.allocate(Math.max(1, c.indexerHeads()));
        this.blockScores = new float[indexedBlocks];

        this.routerScores = ArrayFloatTensor.allocate(c.expertCount());
        this.selectionScores = ArrayFloatTensor.allocate(c.expertCount());
        this.expertIds = new int[c.expertsUsed()];
        this.expertWeights = new float[c.expertsUsed()];
        int width = Math.max(c.expertHiddenDim(), c.sharedHiddenDim());
        this.expertHidden = ArrayFloatTensor.allocate(width);
        this.expertHiddenUp = ArrayFloatTensor.allocate(width);
        this.expertOut = ArrayFloatTensor.allocate(dim);
        clearPending();
    }

    @Override
    protected StateFields createStateFields(Configuration configuration) {
        DeepSeek4Configuration config = (DeepSeek4Configuration) configuration;
        StateFields fields = new StateFields();
        int dim = config.dim();
        fields.x = ArrayFloatTensor.allocate(dim);
        fields.xb = ArrayFloatTensor.allocate(dim);
        fields.xb2 = ArrayFloatTensor.allocate(dim);
        fields.hb = ArrayFloatTensor.allocate(config.expertHiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(config.expertHiddenDim());
        fields.q = ArrayFloatTensor.allocate(config.queryDim());
        fields.k = ArrayFloatTensor.allocate(config.headDim());
        fields.v = ArrayFloatTensor.allocate(config.headDim());
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.maxAttended());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());
        int layers = config.allLayers();
        fields.keyCache = new FloatTensor[layers];
        fields.valueCache = new FloatTensor[layers];
        for (int l = 0; l < layers; l++) {
            fields.keyCache[l] = allocateKeyValue(config.slidingWindow(), config.headDim());
        }
        if (Boolean.parseBoolean(System.getProperty("use.tornadovm", "false"))) {
            allocateDeviceWorkspace(config);
        }
        return fields;
    }

    // ── Device workspace ──────────────────────────────────────────────────────

    // Set while the base constructor runs, so without initializers.
    private int[] pendingOffsets;
    private int[] compressedOffsets;
    private int[] indexerPendingOffsets;
    private int[] indexerKeyOffsets;
    private int[] hashOffsets;

    /** Where layer {@code l}'s window starts in the device window array. */
    public int windowOffset(DeepSeek4Configuration c, int l) {
        return l * c.slidingWindow() * c.headDim();
    }

    /** Where layer {@code l}'s compressor entries start in the device pending arrays. */
    public int pendingOffset(int l) {
        return pendingOffsets[l];
    }

    /** Where layer {@code l}'s compressed keys start. */
    public int compressedOffset(int l) {
        return compressedOffsets[l];
    }

    /** Where layer {@code l}'s indexer entries start in the device pending arrays. */
    public int indexerPendingOffset(int l) {
        return indexerPendingOffsets[l];
    }

    /** Where layer {@code l}'s indexer keys start in the compressed-key array. */
    public int indexerKeyOffset(int l) {
        return indexerKeyOffsets[l];
    }

    /** Where hash-routed layer {@code l}'s experts start in the staged expert array. */
    public int hashOffset(int l) {
        return hashOffsets[l];
    }

    // @formatter:off
    /**
     * The device arrays of the single-token plan, for the trunk (the MTP layer runs on the host).
     * Every compressed layer's pending entries, then every indexer's, share one pair of arrays; the
     * compressed keys and the indexer keys share another.
     */
    // @formatter:on
    private void allocateDeviceWorkspace(DeepSeek4Configuration c) {
        int dim = c.dim();
        int layers = c.numberOfLayers();
        int hd = c.headDim();
        int id = c.indexerHeadDim();
        int moeHidden = Math.max(32, c.expertsUsed() * c.expertHiddenDim() + c.sharedHiddenDim());
        this.pendingOffsets = new int[layers];
        this.compressedOffsets = new int[layers];
        this.indexerPendingOffsets = new int[layers];
        this.indexerKeyOffsets = new int[layers];
        this.hashOffsets = new int[layers];
        int pending = 0;
        int keys = 0;
        int row = 0;
        int maxBlocks = 1;
        for (int l = 0; l < layers; l++) {
            int r = c.ratio(l);
            hashOffsets[l] = Math.min(l, c.hashLayerCount()) * c.expertsUsed();
            if (r == 0) {
                continue;
            }
            int o = DeepSeek4Configuration.overlap(r);
            pendingOffsets[l] = pending;
            pending += o * r * o * hd;
            compressedOffsets[l] = keys;
            keys += c.maxBlocks(r) * hd;
            row = Math.max(row, o * hd);
            if (c.hasIndexer(l)) {
                indexerPendingOffsets[l] = pending;
                pending += 2 * r * 2 * id;
                indexerKeyOffsets[l] = keys;
                keys += c.maxBlocks(r) * id;
                row = Math.max(row, 2 * id);
                maxBlocks = Math.max(maxBlocks, c.maxBlocks(r));
            }
        }
        int widest =
                Math.max(
                        Math.max(dim, c.queryDim()), Math.max(c.groupedOutputDim(), c.qLoraRank()));

        TornadoWorkspaces.activationQ8_0(workspace, dim);
        workspace.wrapX = TornadoWorkspaces.floats(dim);
        workspace.wrapXb = TornadoWorkspaces.floats(dim);
        workspace.wrapXb2 = TornadoWorkspaces.floats(dim);
        TornadoWorkspaces.packedActivation(workspace, widest);
        workspace.wrapHb = TornadoWorkspaces.floats(1);
        workspace.wrapHb2 = TornadoWorkspaces.floats(1);
        workspace.wrapLogits = TornadoWorkspaces.floats(c.vocabularySize());
        workspace.wrapQueryLatent = TornadoWorkspaces.floats(c.qLoraRank());
        workspace.wrapQ = TornadoWorkspaces.floats(c.queryDim());
        workspace.wrapAtt = TornadoWorkspaces.floats(c.numberOfHeads() * c.maxAttended());
        workspace.wrapKeyCache = TornadoWorkspaces.floats(1);
        workspace.wrapValueCache = TornadoWorkspaces.floats(1);
        workspace.wrapRouterLogits = TornadoWorkspaces.floats(c.expertCount());
        workspace.wrapSelectedExperts = TornadoWorkspaces.ints(c.expertsUsed());
        workspace.wrapRoutingWeights = TornadoWorkspaces.floats(c.expertsUsed());
        workspace.wrapMoeHidden = TornadoWorkspaces.floats(moeHidden);
        workspace.wrapMoeHiddenQuants = TornadoWorkspaces.ints(moeHidden / 4);
        workspace.wrapMoeHiddenQScales = TornadoWorkspaces.floats(moeHidden / 32);
        workspace.wrapMoeHiddenQSums = TornadoWorkspaces.ints(moeHidden / 32);

        workspace.deepseek4 =
                DeepSeek4DeviceArrays.allocate(
                        new DeepSeek4DeviceArrays.Sizes(
                                dim,
                                c.hcCount() * dim,
                                c.hcMixCount(),
                                hd,
                                c.queryDim(),
                                c.groupedOutputDim(),
                                layers * c.slidingWindow() * hd,
                                Math.max(1, row),
                                pending,
                                keys,
                                c.indexerHeads() * id,
                                c.indexerHeads(),
                                maxBlocks,
                                Math.min(c.indexerTopK(), maxBlocks),
                                c.hashLayerCount() * c.expertsUsed()),
                        c.ropeFrequencies(false),
                        c.ropeFrequencies(true));

        if (prefillBatchWidth > 1) {
            int capacity = Math.max(1, Math.min(c.indexerTopK(), maxBlocks));
            workspace.deepseek4Batch =
                    DeepSeek4BatchArrays.allocate(
                            new DeepSeek4BatchArrays.Sizes(
                                    (prefillBatchWidth + 127) & ~127,
                                    dim,
                                    c.hcCount() * dim,
                                    c.hcMixCount(),
                                    widest,
                                    c.qLoraRank(),
                                    hd,
                                    c.queryDim(),
                                    c.groupedOutputDim(),
                                    c.numberOfHeads(),
                                    c.maxAttended(),
                                    Math.max(1, row),
                                    c.indexerHeads() * id,
                                    c.indexerHeads(),
                                    maxBlocks,
                                    capacity,
                                    c.expertCount(),
                                    c.expertsUsed(),
                                    moeHidden,
                                    c.hashLayerCount() * c.expertsUsed()));
        }

        // [0] = position, [1] = table-local KV slot.
        workspace.positionHolder = TornadoWorkspaces.ints(2);
        workspace.temp = TornadoWorkspaces.floats(1 + ((dim + localSize - 1) / localSize));
        workspace.tempFFN = TornadoWorkspaces.floats(1 + ((dim + localSize - 1) / localSize));
        workspace.tempLogits = TornadoWorkspaces.floats(1 + ((dim + localSize - 1) / localSize));
    }

    /** Stages the hash-routed layers' experts for {@code token}, from their tables. */
    public void stageHashExperts(int token, int[][] tables, int used, int hashLayers) {
        if (workspace.deepseek4 == null) {
            return;
        }
        for (int l = 0; l < hashLayers; l++) {
            for (int k = 0; k < used; k++) {
                workspace.deepseek4.setHashExpert(l * used + k, tables[l][token * used + k]);
            }
        }
    }

    /** Stages the hash-routed layers' experts for each of a chunk's tokens, row by row. */
    public void stageBatchHashExperts(
            int[] tokens, int chunkSize, int[][] tables, int used, int hashLayers) {
        DeepSeek4BatchArrays b = workspace.deepseek4Batch;
        if (b == null) {
            return;
        }
        for (int t = 0; t < chunkSize; t++) {
            for (int l = 0; l < hashLayers; l++) {
                for (int k = 0; k < used; k++) {
                    b.setHashExpert(t, l * used + k, tables[l][tokens[t] * used + k]);
                }
            }
        }
    }

    /**
     * Makes this state, one stage of a model split across devices, read the hash-routed experts the
     * host stages in {@code session} token by token and chunk by chunk. Where the session has no
     * chunk-wide arrays of its own, it stages into this stage's.
     */
    public void readHashExpertsFrom(DeepSeek4State session) {
        if (workspace.deepseek4 == null || session.workspace.deepseek4 == null) {
            return;
        }
        workspace.deepseek4.readHashExpertsFrom(session.workspace.deepseek4);
        if (workspace.deepseek4Batch != null) {
            if (session.workspace.deepseek4Batch == null) {
                session.workspace.deepseek4Batch = workspace.deepseek4Batch;
            } else {
                workspace.deepseek4Batch.readHashExpertsFrom(session.workspace.deepseek4Batch);
            }
        }
    }

    /** The compressors' pending entries: projections zero, gate scores minus infinity. */
    private void clearPending() {
        for (int l = 0; l < compressorKv.length; l++) {
            clear(compressorKv[l], compressorScore[l]);
            clear(indexerKv[l], indexerScore[l]);
        }
    }

    private static void clear(FloatTensor kv, FloatTensor score) {
        if (kv != null) {
            kv.fillInPlace(0, kv.size(), 0f);
            score.fillInPlace(0, score.size(), Float.NEGATIVE_INFINITY);
        }
    }

    @Override
    public void resetSequenceState() {
        clearPending();
        Arrays.fill(blockScores, 0f);
    }
}
