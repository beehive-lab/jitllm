package org.beehive.jitllm.inference.state;

import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.glm5next.Glm5NextConfiguration;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * A {@code glm5next} session on the host.
 *
 * <p>Three kinds of per-layer memory: the {@link Glm5NextConfiguration#hcCount()} residual streams
 * every layer reads and rewrites; for a KDA layer, its convolution windows and its delta-rule
 * matrices, which {@link #resetSequenceState()} clears; and for a sparse-attention layer (the MTP
 * layer included), one normalized latent per position in {@link #keyCache} plus the indexer's
 * per-position key and pool gate and its per-pool compressed key. The latent is both key and value,
 * so {@link #valueCache} holds nothing.
 */
public final class Glm5NextState extends State {

    /** The residual streams, stream-major: {@code streams[n * dim + i]}. */
    public final FloatTensor streams;

    /** The streams as they entered the current sublayer, mixed back in after it. */
    public final FloatTensor residualStreams;

    /** The mHC coefficients of the current sublayer: collapse, expand and the mix. */
    public final float[] hcMixes;

    public final float[] hcPre;
    public final float[] hcPost;

    /** {@code hcComb[m * n + k]}: how much input stream {@code m} feeds output stream {@code k}. */
    public final float[] hcComb;

    // KDA, per layer (null on sparse-attention layers).
    public final FloatTensor[] convWindowQ;
    public final FloatTensor[] convWindowK;
    public final FloatTensor[] convWindowV;

    /** Per head a {@code headDim × headDim} matrix, row = key channel, column = value channel. */
    public final FloatTensor[] deltaState;

    public final FloatTensor kdaQ;
    public final FloatTensor kdaK;
    public final FloatTensor kdaV;
    public final FloatTensor kdaConvQ;
    public final FloatTensor kdaConvK;
    public final FloatTensor kdaConvV;
    public final FloatTensor kdaLowRank;
    public final FloatTensor kdaDecay;
    public final FloatTensor kdaBeta;
    public final FloatTensor kdaGate;
    public final FloatTensor kdaOut;

    // Sparse attention, per layer (null on KDA layers).
    public final FloatTensor[] indexKeyCache;
    public final FloatTensor[] poolGateCache;
    public final FloatTensor[] poolKeyCache;

    public final FloatTensor queryLatent;
    public final FloatTensor kvLatent;
    public final FloatTensor absorbedQuery;
    public final FloatTensor latentOut;
    public final FloatTensor indexQuery;
    public final FloatTensor indexHeadWeights;
    public final FloatTensor indexKey;
    public final FloatTensor poolGate;
    public final float[] poolScores;

    /** The positions the current query attends to; {@link #selectedCount} of them are valid. */
    public final int[] selected;

    public int selectedCount;

    // Feed-forward.
    public final FloatTensor routerScores;
    public final FloatTensor selectionScores;
    public final int[] expertIds;
    public final float[] expertWeights;
    public final FloatTensor expertHidden;
    public final FloatTensor expertHiddenUp;
    public final FloatTensor expertOut;

    /** The trunk's last normalized hidden state: the MTP layer's second input. */
    public final FloatTensor hidden;

    public final FloatTensor nextnConcat;
    public final FloatTensor nextnX;
    public final FloatTensor nextnLogits;

    public Glm5NextState(Configuration config, int batchsize) {
        this(config, batchsize, null);
    }

    public Glm5NextState(
            Configuration config, int batchsize, org.beehive.jitllm.runtime.kv.KvLease lease) {
        super(config, batchsize, lease);
        Glm5NextConfiguration c = (Glm5NextConfiguration) config;
        int dim = c.dim();
        int n = c.hcCount();
        int layers = c.allLayers();
        int context = c.contextLength();
        int pools = context / c.indexerPoolSize() + 1;

        this.streams = ArrayFloatTensor.allocate(n * dim);
        this.residualStreams = ArrayFloatTensor.allocate(n * dim);
        this.hcMixes = new float[c.hcMixCount()];
        this.hcPre = new float[n];
        this.hcPost = new float[n];
        this.hcComb = new float[n * n];

        int kda = c.kdaDim();
        int window = kda * (c.kdaConvKernel() - 1);
        this.convWindowQ = new FloatTensor[layers];
        this.convWindowK = new FloatTensor[layers];
        this.convWindowV = new FloatTensor[layers];
        this.deltaState = new FloatTensor[layers];
        this.indexKeyCache = new FloatTensor[layers];
        this.poolGateCache = new FloatTensor[layers];
        this.poolKeyCache = new FloatTensor[layers];
        for (int l = 0; l < layers; l++) {
            if (c.isKda(l)) {
                convWindowQ[l] = ArrayFloatTensor.allocate(window);
                convWindowK[l] = ArrayFloatTensor.allocate(window);
                convWindowV[l] = ArrayFloatTensor.allocate(window);
                deltaState[l] =
                        ArrayFloatTensor.allocate(c.kdaHeads() * c.kdaHeadDim() * c.kdaHeadDim());
            } else {
                indexKeyCache[l] = ArrayFloatTensor.allocate(context * c.indexerHeadDim());
                poolGateCache[l] = ArrayFloatTensor.allocate(context * c.indexerHeadDim());
                poolKeyCache[l] = ArrayFloatTensor.allocate(pools * c.indexerHeadDim());
            }
        }
        this.kdaQ = ArrayFloatTensor.allocate(kda);
        this.kdaK = ArrayFloatTensor.allocate(kda);
        this.kdaV = ArrayFloatTensor.allocate(kda);
        this.kdaConvQ = ArrayFloatTensor.allocate(kda);
        this.kdaConvK = ArrayFloatTensor.allocate(kda);
        this.kdaConvV = ArrayFloatTensor.allocate(kda);
        this.kdaLowRank = ArrayFloatTensor.allocate(c.kdaHeadDim());
        this.kdaDecay = ArrayFloatTensor.allocate(kda);
        this.kdaBeta = ArrayFloatTensor.allocate(c.kdaHeads());
        this.kdaGate = ArrayFloatTensor.allocate(kda);
        this.kdaOut = ArrayFloatTensor.allocate(kda);

        this.queryLatent = ArrayFloatTensor.allocate(c.qLoraRank());
        this.kvLatent = ArrayFloatTensor.allocate(c.kvLoraRank());
        this.absorbedQuery = ArrayFloatTensor.allocate(c.absorbedQueryDim());
        this.latentOut = ArrayFloatTensor.allocate(c.absorbedQueryDim());
        this.indexQuery = ArrayFloatTensor.allocate(c.indexerHeads() * c.indexerHeadDim());
        this.indexHeadWeights = ArrayFloatTensor.allocate(c.indexerHeads());
        this.indexKey = ArrayFloatTensor.allocate(c.indexerHeadDim());
        this.poolGate = ArrayFloatTensor.allocate(c.indexerHeadDim());
        this.poolScores = new float[pools];
        this.selected = new int[context + c.indexerPoolSize()];

        this.routerScores = ArrayFloatTensor.allocate(Math.max(1, c.expertCount()));
        this.selectionScores = ArrayFloatTensor.allocate(Math.max(1, c.expertCount()));
        this.expertIds = new int[Math.max(1, c.expertsUsed())];
        this.expertWeights = new float[Math.max(1, c.expertsUsed())];
        int width = Math.max(1, Math.max(c.expertHiddenDim(), c.sharedHiddenDim()));
        this.expertHidden = ArrayFloatTensor.allocate(width);
        this.expertHiddenUp = ArrayFloatTensor.allocate(width);
        this.expertOut = ArrayFloatTensor.allocate(dim);

        this.hidden = ArrayFloatTensor.allocate(dim);
        this.nextnConcat = ArrayFloatTensor.allocate(2 * dim);
        this.nextnX = ArrayFloatTensor.allocate(dim);
        this.nextnLogits = ArrayFloatTensor.allocate(c.vocabularySize());
    }

    @Override
    protected StateFields createStateFields(Configuration configuration) {
        Glm5NextConfiguration config = (Glm5NextConfiguration) configuration;
        StateFields fields = new StateFields();
        int dim = config.dim();
        int hidden = Math.max(config.hiddenDim(), 1);
        fields.x = ArrayFloatTensor.allocate(dim);
        fields.xb =
                ArrayFloatTensor.allocate(
                        Math.max(dim, Math.max(config.attentionOutputInputDim(), config.kdaDim())));
        fields.xb2 = ArrayFloatTensor.allocate(dim);
        fields.hb = ArrayFloatTensor.allocate(hidden);
        fields.hb2 = ArrayFloatTensor.allocate(hidden);
        fields.q = ArrayFloatTensor.allocate(config.queryDim());
        fields.k = ArrayFloatTensor.allocate(config.kvLoraRank());
        fields.v = ArrayFloatTensor.allocate(config.kvLoraRank());
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());
        int layers = config.allLayers();
        fields.keyCache = new FloatTensor[layers];
        fields.valueCache = new FloatTensor[layers];
        for (int l = 0; l < layers; l++) {
            if (!config.isKda(l)) {
                fields.keyCache[l] = allocateKeyValue(config.contextLength(), config.kvLoraRank());
            }
        }
        return fields;
    }

    /**
     * Clears the convolution windows and delta-rule matrices. The attention caches are left alone:
     * a query reads only positions up to its own, so rewinding the position is enough for them.
     */
    @Override
    public void resetSequenceState() {
        for (int l = 0; l < deltaState.length; l++) {
            if (deltaState[l] != null) {
                convWindowQ[l].fillInPlace(0, convWindowQ[l].size(), 0f);
                convWindowK[l].fillInPlace(0, convWindowK[l].size(), 0f);
                convWindowV[l].fillInPlace(0, convWindowV[l].size(), 0f);
                deltaState[l].fillInPlace(0, deltaState[l].size(), 0f);
            }
        }
    }
}
