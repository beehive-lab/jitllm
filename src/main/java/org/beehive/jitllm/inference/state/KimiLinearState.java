package org.beehive.jitllm.inference.state;

import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.kimilinear.KimiLinearConfiguration;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * A {@code kimi-linear} session on the host.
 *
 * <p>A KDA layer keeps its convolution windows and delta-rule matrices in {@link #kda}, which
 * {@link #resetSequenceState()} clears. An MLA layer keeps one {@link
 * KimiLinearConfiguration#keyWidth()}-wide row per position in {@link #keyCache}: the normalized
 * latent, then the unrotated key tail. The latent half is also the value, so {@link #valueCache}
 * holds nothing.
 */
public final class KimiLinearState extends State {

    /** KDA's windows, matrices and per-token buffers (null slots on MLA layers). */
    public final KdaHostState kda;

    /** {@code attn_kv_a_mqa}'s output: the latent, then the key tail. */
    public final FloatTensor compressedKeyValue;

    /** Every head's absorbed query: the latent-space no-rope part, then the tail. */
    public final FloatTensor absorbedQuery;

    /** Every head's attended latent, before {@code attn_v_b}. */
    public final FloatTensor latentOut;

    public final FloatTensor routerScores;
    public final FloatTensor selectionScores;
    public final int[] expertIds;
    public final float[] expertWeights;
    public final FloatTensor expertHidden;
    public final FloatTensor expertHiddenUp;
    public final FloatTensor expertOut;

    public KimiLinearState(Configuration config, int batchsize) {
        this(config, batchsize, null);
    }

    public KimiLinearState(
            Configuration config, int batchsize, org.beehive.jitllm.runtime.kv.KvLease lease) {
        super(config, batchsize, lease);
        KimiLinearConfiguration c = (KimiLinearConfiguration) config;
        this.kda =
                new KdaHostState(
                        c.numberOfLayers(),
                        c::isKda,
                        c.kdaHeads(),
                        c.kdaHeadDim(),
                        c.kdaConvKernel());
        this.compressedKeyValue = ArrayFloatTensor.allocate(c.compressedKeyValueDim());
        this.absorbedQuery = ArrayFloatTensor.allocate(c.absorbedQueryDim());
        this.latentOut = ArrayFloatTensor.allocate(c.numberOfHeads() * c.kvLoraRank());
        this.routerScores = ArrayFloatTensor.allocate(Math.max(1, c.expertCount()));
        this.selectionScores = ArrayFloatTensor.allocate(Math.max(1, c.expertCount()));
        this.expertIds = new int[Math.max(1, c.expertsUsed())];
        this.expertWeights = new float[Math.max(1, c.expertsUsed())];
        int width = Math.max(1, Math.max(c.expertHiddenDim(), c.sharedHiddenDim()));
        this.expertHidden = ArrayFloatTensor.allocate(width);
        this.expertHiddenUp = ArrayFloatTensor.allocate(width);
        this.expertOut = ArrayFloatTensor.allocate(c.dim());
    }

    @Override
    protected int batchQDim(Configuration config) {
        return ((KimiLinearConfiguration) config).absorbedQueryDim();
    }

    @Override
    protected int batchKvDim(Configuration config) {
        return ((KimiLinearConfiguration) config).keyWidth();
    }

    @Override
    protected StateFields createStateFields(Configuration configuration) {
        KimiLinearConfiguration config = (KimiLinearConfiguration) configuration;
        StateFields fields = new StateFields();
        int dim = config.dim();
        int hidden = Math.max(config.hiddenDim(), config.expertHiddenDim());
        fields.x = ArrayFloatTensor.allocate(dim);
        fields.xb =
                ArrayFloatTensor.allocate(
                        Math.max(dim, Math.max(config.attentionOutputInputDim(), config.kdaDim())));
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
            if (!config.isKda(l)) {
                // The host cache only where the host runs; a device session keeps its own.
                fields.keyCache[l] =
                        allocateKeyValue(device ? 1 : config.contextLength(), config.keyWidth());
            }
        }
        if (device) {
            KimiLinearDeviceWorkspace.allocate(this, config);
        }
        return fields;
    }

    /** The KDA windows and matrices, which a reset uploads after zeroing. */
    @Override
    public Object[] recurrentDeviceBuffers() {
        if (workspace.wrapConvState == null || workspace.wrapDeltaState == null) {
            return new Object[0];
        }
        return new Object[] {workspace.wrapConvState, workspace.wrapDeltaState};
    }

    /**
     * Clears the convolution windows and delta-rule matrices. The latent caches are left alone: a
     * query reads only positions up to its own, so rewinding the position is enough for them.
     */
    @Override
    public void resetSequenceState() {
        if (kda != null) {
            kda.reset();
        }
        org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces.zeroRecurrentState(
                workspace);
    }
}
