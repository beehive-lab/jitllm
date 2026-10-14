package org.beehive.jitllm.inference.state;

import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.lfm2.Lfm2Configuration;
import org.beehive.jitllm.runtime.kv.KvLease;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * Session state for the {@code lfm2moe} architecture, on the host.
 *
 * <p>Like {@link Qwen35State}, it holds key/value storage only for the layers that attend — ten of
 * the 24B's forty — and a fixed-size recurrent state for the rest: each short-convolution layer
 * keeps the last {@code kernel - 1} values of its gated input per channel. {@link #keyCache} and
 * {@link #valueCache} are {@code null} at a convolution layer's index, {@link #convState} at an
 * attention layer's.
 *
 * <p>The convolution window has no position mask, so {@link #resetSequenceState()} has to clear it:
 * a stale window conditions the next sequence's first two tokens on the previous sequence's last
 * two.
 */
public final class Lfm2State extends State {

    /**
     * Rolling convolution history per convolution layer, {@code dim * (kernel - 1)} elements,
     * channel-major and oldest tap first, as {@code CpuOperations.causalConv1d} reads it. {@code
     * null} at an attention layer.
     */
    public final FloatTensor[] convState;

    /** {@code in_proj}'s output: {@code B ‖ C ‖ x}, {@code 3 * dim}. */
    public final FloatTensor convBcx;

    /** The gated input {@code B ⊙ x} the convolution runs over, {@code dim}. */
    public final FloatTensor convBx;

    /** The convolution's output, gated by {@code C} in place before the output projection. */
    public final FloatTensor convOut;

    /** The router's per-expert scores. */
    public final FloatTensor routerScores;

    /** The selected experts, in selection order. */
    public final int[] selectedExperts;

    /** Their mixing weights, normalized to sum to one. */
    public final float[] selectedExpertWeights;

    /** One expert's gate and up activations. */
    public final FloatTensor hbE;

    public final FloatTensor hbE2;

    /** One expert's output, before it is weighted into the residual stream. */
    public final FloatTensor yTmp;

    public Lfm2State(Configuration config, int batchsize) {
        this(config, batchsize, null);
    }

    public Lfm2State(Configuration config, int batchsize, KvLease lease) {
        super(config, batchsize, lease);
        Lfm2Configuration c = (Lfm2Configuration) config;
        this.convState = new FloatTensor[c.numberOfLayers()];
        for (int l = 0; l < c.numberOfLayers(); l++) {
            if (!c.isAttentionLayer(l)) {
                this.convState[l] = ArrayFloatTensor.allocate(c.convStateSize());
            }
        }
        this.convBcx = ArrayFloatTensor.allocate(c.shortConvProjectionDim());
        this.convBx = ArrayFloatTensor.allocate(c.dim());
        this.convOut = ArrayFloatTensor.allocate(c.dim());
        this.routerScores = ArrayFloatTensor.allocate(c.expertCount());
        this.selectedExperts = new int[c.expertUsedCount()];
        this.selectedExpertWeights = new float[c.expertUsedCount()];
        this.hbE = ArrayFloatTensor.allocate(c.expertHiddenDim());
        this.hbE2 = ArrayFloatTensor.allocate(c.expertHiddenDim());
        this.yTmp = ArrayFloatTensor.allocate(c.dim());
    }

    /**
     * Whether this session is being built for a device, handed in for one construction.
     *
     * <p>The same mechanism as {@code Qwen35State.withDeviceArrays}, for the same reason: {@code
     * createStateFields} runs inside the {@code State} constructor, before any field of this class
     * is assigned. The model decides, from whether its weights are device weights.
     */
    private static final ThreadLocal<Boolean> DEVICE_FOR_CONSTRUCTION = new ThreadLocal<>();

    /** Builds a state that allocates its device arrays, or does not. */
    public static <T> T withDeviceArrays(boolean device, java.util.function.Supplier<T> build) {
        Boolean previous = DEVICE_FOR_CONSTRUCTION.get();
        DEVICE_FOR_CONSTRUCTION.set(device);
        try {
            return build.get();
        } finally {
            if (previous == null) {
                DEVICE_FOR_CONSTRUCTION.remove();
            } else {
                DEVICE_FOR_CONSTRUCTION.set(previous);
            }
        }
    }

    /**
     * Zeroes the convolution windows, on the host and on the device; the key/value caches are
     * masked by position instead.
     */
    @Override
    public void resetSequenceState() {
        for (FloatTensor window : convState) {
            if (window != null) {
                window.fillInPlace(0, window.size(), 0f);
            }
        }
        TornadoWorkspaces.zeroRecurrentState(workspace);
    }

    /** The device convolution windows, which the plan re-uploads after a reset. */
    @Override
    public Object[] recurrentDeviceBuffers() {
        return workspace.wrapConvState == null
                ? new Object[0]
                : new Object[] {workspace.wrapConvState};
    }

    @Override
    protected int batchKvDim(Configuration config) {
        return config.kvDim();
    }

    @Override
    protected StateFields createStateFields(Configuration configuration) {
        Lfm2Configuration config = (Lfm2Configuration) configuration;
        StateFields fields = new StateFields();
        int kvDim = config.kvDim();

        fields.x = ArrayFloatTensor.allocate(config.dim());
        fields.xb = ArrayFloatTensor.allocate(config.dim());
        fields.xb2 = ArrayFloatTensor.allocate(config.dim());
        fields.hb = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.q = ArrayFloatTensor.allocate(config.dim());
        fields.k = ArrayFloatTensor.allocate(kvDim);
        fields.v = ArrayFloatTensor.allocate(kvDim);
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());

        int layers = config.numberOfLayers();
        fields.keyCache = new FloatTensor[layers];
        fields.valueCache = new FloatTensor[layers];
        for (int l = 0; l < layers; l++) {
            if (config.isAttentionLayer(l)) {
                fields.keyCache[l] = allocateKeyValue(config.contextLength(), kvDim);
                fields.valueCache[l] = allocateKeyValue(config.contextLength(), kvDim);
            }
        }
        if (Boolean.TRUE.equals(DEVICE_FOR_CONSTRUCTION.get())) {
            allocateDeviceWorkspace(config, fields, kvDim);
        } else {
            fields.kvBlockCfg = 0;
            fields.kvBlockStride = 0;
        }
        return fields;
    }

    // @formatter:off
    /**
     * The device arrays, when this session was built for an accelerator.
     *
     * <ul>
     *   <li><b>Key/value storage covers the attending layers only</b>, addressed by their dense
     *       index — ten of forty in the 24B.
     *   <li><b>The convolution windows are one zeroed array</b>, a slice per convolution layer,
     *       updated in place and never read back.
     *   <li><b>The expert hidden activations reuse {@code wrapHb}</b>: the selected experts' {@code
     *       used × expertHidden} fit inside the dense layers' {@code hiddenDim}, and a layer is one
     *       or the other.
     * </ul>
     */
    // @formatter:on
    private void allocateDeviceWorkspace(Lfm2Configuration config, StateFields fields, int kvDim) {
        switch (config.quantization()) {
            case "FP16" -> TornadoWorkspaces.activationFP16(workspace, config.dim());
            case "Q8_0" -> TornadoWorkspaces.activationQ8_0(workspace, config.dim());
            default ->
                    throw new UnsupportedOperationException(
                            "Unsupported quantization format: " + config.quantization());
        }
        int hidden =
                Math.max(config.hiddenDim(), config.expertUsedCount() * config.expertHiddenDim());
        workspace.wrapX = TornadoWorkspaces.floats(config.dim());
        workspace.wrapXb = TornadoWorkspaces.floats(config.dim());
        workspace.wrapXb2 = TornadoWorkspaces.floats(config.dim());
        workspace.wrapHb = TornadoWorkspaces.floats(hidden);
        workspace.wrapHb2 = TornadoWorkspaces.floats(hidden);
        workspace.wrapLogits = TornadoWorkspaces.floats(config.vocabularySize());
        workspace.wrapQ = TornadoWorkspaces.floats(config.dim());
        workspace.wrapK = TornadoWorkspaces.floats(kvDim);
        workspace.wrapV = TornadoWorkspaces.floats(kvDim);
        workspace.wrapAtt =
                TornadoWorkspaces.floats(config.numberOfHeads() * config.contextLength());

        workspace.wrapSsmQkv = TornadoWorkspaces.floats(config.shortConvProjectionDim());
        workspace.wrapSsmConvOut = TornadoWorkspaces.floats(config.dim());
        workspace.wrapConvState =
                TornadoWorkspaces.zeroedFloats(
                        Math.max(1, config.convLayerCount() * config.convStateSize()));

        workspace.wrapRouterLogits = TornadoWorkspaces.floats(config.expertCount());
        workspace.wrapSelectedExperts = TornadoWorkspaces.ints(config.expertUsedCount());
        workspace.wrapSelectedExpertWeights = TornadoWorkspaces.floats(config.expertUsedCount());

        // [0] = position, [1] = table-local KV slot.
        workspace.positionHolder = TornadoWorkspaces.ints(2);
        workspace.temp = TornadoWorkspaces.floats(1 + ((config.dim() + localSize - 1) / localSize));
        workspace.tempFFN =
                TornadoWorkspaces.floats(1 + ((config.dim() + localSize - 1) / localSize));
        workspace.tempLogits =
                TornadoWorkspaces.floats(1 + ((config.dim() + localSize - 1) / localSize));

        fillKvFields(fields, config, kvDim, config.keyValueLayerCount(), true);
    }
}
