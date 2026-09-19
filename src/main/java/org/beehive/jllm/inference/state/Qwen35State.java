package org.beehive.jllm.inference.state;

import org.beehive.jllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.qwen35.Qwen35Configuration;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tensor.standard.FloatTensor;

/**
 * Session state for the {@code qwen35} architecture.
 *
 * <p>Two things distinguish it from every other family's state.
 *
 * <h2>Key/value storage exists for a quarter of the layers</h2>
 *
 * <p>Only the attention layers have keys and values to retain. The 48 recurrent layers of the 27B
 * would otherwise each be handed a full context-length cache they never write — at 8k context and
 * this model's 1024-wide KV that is 2.5 GB of untouched arrays. {@link #keyCache} and {@link
 * #valueCache} are therefore {@code null} at a recurrent layer's index, and the arrays stay indexed
 * by absolute layer so no caller keeps a second numbering.
 *
 * <h2>Recurrent layers hold state instead, and it is not a cache</h2>
 *
 * <p>{@link #convState} and {@link #deltaState} are fixed-size per layer and independent of
 * position: a delta-net layer's entire history is summed into a {@code head_v_dim × head_v_dim}
 * matrix per value head, and the convolution keeps the last {@code kernel - 1} inputs. That is why
 * they are session state and not KV storage — nothing about them can be paged, evicted, shared or
 * leased, and the engine's cache manager would have nothing to manage.
 *
 * <p>It is also why {@link #resetSequenceState()} matters here and is a no-op elsewhere. A
 * key/value cache does not need clearing between sequences because attention only reads up to the
 * current position; a recurrent state has no such mask, and a stale one silently conditions the new
 * sequence on the old one.
 */
public final class Qwen35State extends State {

    /**
     * Rolling convolution history per recurrent layer, {@code convDim * (kernel - 1)} elements,
     * channel-major and oldest tap first: element {@code c * (kernel - 1) + t} is channel {@code c}
     * as it stood {@code kernel - 1 - t} steps ago. Channel-major to match how GGUF lays out {@code
     * ssm_conv1d}, so the convolution reads both operands the same way. {@code null} at an
     * attention layer.
     */
    public final FloatTensor[] convState;

    /**
     * Delta-net state per recurrent layer: {@code numberOfValueHeads} matrices of {@code
     * headValueDim × headValueDim}, indexed {@code (h * S + i) * S + j} for key row {@code i} and
     * value column {@code j}. {@code null} at an attention layer.
     */
    public final FloatTensor[] deltaState;

    // ---- scratch, sized once ------------------------------------------------

    /** The fused {@code q ‖ k ‖ v} projection of a recurrent layer, before the convolution. */
    public final FloatTensor ssmQkv;

    /** The same after the depthwise convolution and its SiLU. */
    public final FloatTensor ssmConvOut;

    /** The {@code z} gate the delta-net output is normalized against. */
    public final FloatTensor ssmZ;

    /**
     * Per-value-head decay, from {@code ssm_alpha} through its bias, softplus and {@code ssm_a}.
     */
    public final FloatTensor ssmAlpha;

    /** Per-value-head write strength, from {@code ssm_beta} through the logistic. */
    public final FloatTensor ssmBeta;

    /** The recurrent branch's readout, {@code value_dim} wide, before the gated norm. */
    public final FloatTensor ssmOut;

    /** The gated norm's result, feeding the recurrent output projection. */
    public final FloatTensor ssmNormed;

    /**
     * The trunk's final hidden state, after {@code output_norm} and before the LM head.
     *
     * <p>Kept because the MTP block consumes it: its input is this vector concatenated with the
     * drafted token's embedding. Written by every trunk forward pass whether or not speculation is
     * running, because the cost is one copy of {@code dim} floats and the alternative is a forward
     * pass whose behaviour depends on a mode flag.
     */
    public final FloatTensor hNextn;

    /** The MTP block's concatenated {@code [enorm(embedding) ‖ hnorm(hidden)]}, {@code 2 * dim}. */
    public final FloatTensor nextnConcat;

    /**
     * The MTP block's residual stream, kept apart from {@link #x} so a draft step does not disturb
     * the trunk's. The two are alive at the same time: the draft is computed from the trunk's
     * hidden state while the trunk is between steps.
     */
    public final FloatTensor nextnX;

    /**
     * The draft head's logits, kept apart from {@link #logits} so drafting cannot overwrite the
     * trunk's prediction. The speculative loop holds both at once — the trunk's, to sample the
     * token being committed, and the draft's, to guess the one after it.
     */
    public final FloatTensor nextnLogits;

    /** The query half of an attention layer's fused query/gate projection, de-interleaved. */
    public final FloatTensor attnQ;

    /** Its gate half. Applied through a logistic to the attention result. */
    public final FloatTensor attnGate;

    /** The convolved query slice of a recurrent layer, {@code keyHeads * headKeyDim}. */
    public final FloatTensor ssmQ;

    /** Its key slice, the same width. */
    public final FloatTensor ssmK;

    /** Its value slice, {@code valueHeads * headValueDim}. */
    public final FloatTensor ssmV;

    public Qwen35State(Configuration config, int batchsize) {
        this(config, batchsize, null);
    }

    public Qwen35State(
            Configuration config, int batchsize, org.beehive.jllm.runtime.kv.KvLease lease) {
        super(config, batchsize, lease);
        Qwen35Configuration c = (Qwen35Configuration) config;

        this.convState = new FloatTensor[c.numberOfLayers()];
        this.deltaState = new FloatTensor[c.numberOfLayers()];
        for (int l = 0; l < c.numberOfLayers(); l++) {
            if (c.isRecurrentLayer(l)) {
                this.convState[l] = ArrayFloatTensor.allocate(c.convStateSize());
                this.deltaState[l] = ArrayFloatTensor.allocate(c.deltaNetStateSize());
            }
        }

        this.ssmQkv = ArrayFloatTensor.allocate(c.deltaNetConvDim());
        this.ssmConvOut = ArrayFloatTensor.allocate(c.deltaNetConvDim());
        this.ssmZ = ArrayFloatTensor.allocate(c.deltaNetValueDim());
        this.ssmAlpha = ArrayFloatTensor.allocate(c.numberOfValueHeads());
        this.ssmBeta = ArrayFloatTensor.allocate(c.numberOfValueHeads());
        this.ssmOut = ArrayFloatTensor.allocate(c.deltaNetValueDim());
        this.ssmNormed = ArrayFloatTensor.allocate(c.deltaNetValueDim());
        this.hNextn = ArrayFloatTensor.allocate(c.dim());
        this.nextnConcat = ArrayFloatTensor.allocate(2 * c.dim());
        this.nextnX = ArrayFloatTensor.allocate(c.dim());
        this.nextnLogits = ArrayFloatTensor.allocate(c.vocabularySize());
        this.attnQ = ArrayFloatTensor.allocate(c.attentionOutputInputDim());
        this.attnGate = ArrayFloatTensor.allocate(c.attentionOutputInputDim());
        this.ssmQ = ArrayFloatTensor.allocate(c.deltaNetKeyDim());
        this.ssmK = ArrayFloatTensor.allocate(c.deltaNetKeyDim());
        this.ssmV = ArrayFloatTensor.allocate(c.deltaNetValueDim());
    }

    /**
     * Whether this session is being built for a device, handed in for one construction.
     *
     * <p>The same shape as {@code State.withStorageOptions}, and for the same mechanical reason:
     * {@code createStateFields} runs inside the {@code State} constructor, so a subclass field
     * assigned afterwards is too late to be read by it.
     */
    private static final ThreadLocal<Boolean> DEVICE_FOR_CONSTRUCTION = new ThreadLocal<>();

    /**
     * Builds a state that allocates its device arrays, or does not.
     *
     * @param device whether a device plan will be built for this session — in practice, whether the
     *     model's weights are device weights
     */
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

    // @formatter:off
    /**
     * Whether the device arrays are worth allocating for this session.
     *
     * <p>Answered by the caller that knows — the model, from whether its weights are device weights
     * — and only otherwise from the property the facade defaults its backend from. The property
     * alone was wrong as soon as a plan provider existed: a caller that loads device weights
     * without setting it got a session whose device arrays were all null, and the failure arrived
     * from inside TornadoVM as {@code null object passed into streamIn()} in the activation graph
     * rather than anywhere that named the cause.
     *
     * <p>Why gate at all, when every other family allocates unconditionally: for them the waste is
     * a few megabytes, and here it is over a gigabyte — 151 MB of recurrent state and the key/value
     * store — on a host path that has already allocated its own.
     */
    // @formatter:on
    private static boolean deviceInPlay() {
        Boolean handedIn = DEVICE_FOR_CONSTRUCTION.get();
        return handedIn != null
                ? handedIn
                : Boolean.parseBoolean(System.getProperty("use.tornadovm", "false"));
    }

    /**
     * Zeroes the recurrent state so a reused session does not continue the previous sequence.
     *
     * <p>The key/value caches are deliberately left alone: attention reads only up to the current
     * position, so rewinding the position is enough for them, and clearing them would cost a
     * context-length write for no effect.
     */
    @Override
    public void resetSequenceState() {
        for (int l = 0; l < convState.length; l++) {
            if (convState[l] != null) {
                convState[l].fillInPlace(0, convState[l].size(), 0f);
                deltaState[l].fillInPlace(0, deltaState[l].size(), 0f);
            }
        }
        // The device arrays too, when there are any. They are the same state in the other
        // representation, and clearing one and not the other would leave a reset session correct on
        // whichever path this reset happened to be thinking about.
        TornadoWorkspaces.zeroRecurrentState(workspace);
    }

    /**
     * The convolution windows and the delta-net matrices, which the graphs keep on the device.
     *
     * <p>Empty on a host-only session, which allocates neither.
     */
    @Override
    public Object[] recurrentDeviceBuffers() {
        if (workspace.wrapConvState == null || workspace.wrapDeltaState == null) {
            return new Object[0];
        }
        return new Object[] {workspace.wrapConvState, workspace.wrapDeltaState};
    }

    @Override
    protected int batchQDim(Configuration config) {
        return ((Qwen35Configuration) config).attentionOutputInputDim();
    }

    @Override
    protected int batchKvDim(Configuration config) {
        return ((Qwen35Configuration) config).kvDim();
    }

    @Override
    protected StateFields createStateFields(Configuration configuration) {
        Qwen35Configuration config = (Qwen35Configuration) configuration;
        StateFields fields = new StateFields();

        int kvDim = config.kvDim();

        fields.x = ArrayFloatTensor.allocate(config.dim());
        // Wide enough for the attention branch's concatenated heads, which exceed dim here
        // (24 heads of 256 against a 5120 embedding), and reused by the feed-forward branch.
        fields.xb =
                ArrayFloatTensor.allocate(Math.max(config.attentionOutputInputDim(), config.dim()));
        fields.xb2 = ArrayFloatTensor.allocate(config.dim());
        fields.hb = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(config.hiddenDim());
        // Query and output gate arrive fused, interleaved per head, and stay that way.
        fields.q = ArrayFloatTensor.allocate(config.queryGateDim());
        fields.k = ArrayFloatTensor.allocate(kvDim);
        fields.v = ArrayFloatTensor.allocate(kvDim);
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());

        // One cache per block that actually attends; null elsewhere, indexed by absolute block.
        int blocks = config.numberOfBlocks();
        fields.keyCache = new FloatTensor[blocks];
        fields.valueCache = new FloatTensor[blocks];
        for (int l = 0; l < blocks; l++) {
            if (!config.isRecurrentLayer(l)) {
                fields.keyCache[l] = ArrayFloatTensor.allocate(config.contextLength(), kvDim);
                fields.valueCache[l] = ArrayFloatTensor.allocate(config.contextLength(), kvDim);
            }
        }

        allocateDeviceWorkspace(config, fields, kvDim);
        return fields;
    }

    // @formatter:off
    /**
     * The device arrays, when this session was built for an accelerator.
     *
     * <p>Skipped entirely on the host path. Nothing here is small — the recurrent state alone is
     * 151 MB at the 27B's shape — and a CPU session that allocated it would reserve memory it never
     * touches, on a model that has little to spare.
     *
     * <p>Three things about this family's device memory are unlike every other family's.
     *
     * <ul>
     *   <li><b>Key/value storage covers a quarter of the blocks.</b> Only the attending layers
     *       write to it, and they address it by a dense index. Sizing it by the layer count would
     *       cost four times as much for nothing.
     *   <li><b>The recurrent layers hold state that is neither cache nor scratch.</b> Convolution
     *       windows and delta-net matrices persist across tokens and are updated in place. One
     *       array per kind, addressed by a per-layer offset, because 48 buffers per kind would be
     *       48 transfers to arrange and keep resident.
     *   <li><b>That state must start at zero.</b> A key/value cache need not — attention reads no
     *       further than the current position — but a recurrence has no such mask, and whatever
     *       happened to be in the allocation would be read as the sequence's own history.
     * </ul>
     */
    // @formatter:on
    private void allocateDeviceWorkspace(
            Qwen35Configuration config, StateFields fields, int kvDim) {
        if (!deviceInPlay()) {
            // Host-only session: the plan that would read these is never built.
            fields.kvBlockCfg = 0;
            fields.kvBlockStride = 0;
            return;
        }

        switch (config.quantization()) {
            case "FP16" -> TornadoWorkspaces.activationFP16(workspace, config.dim());
            case "Q8_0" -> TornadoWorkspaces.activationQ8_0(workspace, config.dim());
            default ->
                    throw new UnsupportedOperationException(
                            "Unsupported quantization format: " + config.quantization());
        }

        int queryDim = config.attentionOutputInputDim();
        workspace.wrapX = TornadoWorkspaces.floats(config.dim());
        // Wide enough for the attention branch's concatenated heads, which exceed dim here, and
        // reused by the feed-forward branch and by the delta-net branch's normalized input.
        workspace.wrapXb = TornadoWorkspaces.floats(Math.max(queryDim, config.dim()));
        workspace.wrapXb2 = TornadoWorkspaces.floats(config.dim());
        // An activation in Q8 blocks, for the packed-integer projections: four quants per int, one
        // scale and one sum of quants per block of 32. Sized for the widest activation any of them
        // reads -- the feed-forward's hidden width -- and used as a prefix by the narrower ones,
        // so the three arrays serve every projection rather than one set per width.
        int widest = Math.max(config.dim(), config.hiddenDim());
        workspace.wrapXbQuants = TornadoWorkspaces.ints(widest / 4);
        workspace.wrapXbScales = TornadoWorkspaces.floats(widest / 32);
        workspace.wrapXbSums = TornadoWorkspaces.ints(widest / 32);
        workspace.wrapHb = TornadoWorkspaces.floats(config.hiddenDim());
        workspace.wrapHb2 = TornadoWorkspaces.floats(config.hiddenDim());
        workspace.wrapLogits = TornadoWorkspaces.floats(config.vocabularySize());

        // The fused query/gate projection, and the two halves it separates into.
        workspace.wrapQ = TornadoWorkspaces.floats(config.queryGateDim());
        workspace.wrapAttnQ = TornadoWorkspaces.floats(queryDim);
        workspace.wrapAttnGate = TornadoWorkspaces.floats(queryDim);
        workspace.wrapK = TornadoWorkspaces.floats(kvDim);
        workspace.wrapV = TornadoWorkspaces.floats(kvDim);
        workspace.wrapAtt =
                TornadoWorkspaces.floats(config.numberOfHeads() * config.contextLength());
        // The split-KV scratch, in its own buffer rather than sharing wrapAtt: per head, nSplits
        // partial numerators of headSize, then nSplits maxima and nSplits sums. This family's
        // attention head is config.headSize() wide, which is not the delta-net value head the
        // recurrent branch is sized by.
        workspace.wrapAttSplit =
                TornadoWorkspaces.floats(
                        config.numberOfHeads()
                                * Qwen35Configuration.DECODE_ATTENTION_SPLITS
                                * (config.headSize() + 2));

        // The delta-net branch's scratch.
        workspace.wrapSsmQkv = TornadoWorkspaces.floats(config.deltaNetConvDim());
        workspace.wrapSsmConvOut = TornadoWorkspaces.floats(config.deltaNetConvDim());
        workspace.wrapSsmZ = TornadoWorkspaces.floats(config.deltaNetValueDim());
        workspace.wrapSsmAlpha = TornadoWorkspaces.floats(config.numberOfValueHeads());
        workspace.wrapSsmBeta = TornadoWorkspaces.floats(config.numberOfValueHeads());
        workspace.wrapSsmQ = TornadoWorkspaces.floats(config.deltaNetKeyDim());
        workspace.wrapSsmK = TornadoWorkspaces.floats(config.deltaNetKeyDim());
        workspace.wrapSsmV = TornadoWorkspaces.floats(config.deltaNetValueDim());
        workspace.wrapSsmOut = TornadoWorkspaces.floats(config.deltaNetValueDim());

        // The recurrent state, zeroed: see the note above on why this is not optional.
        int recurrent = config.recurrentLayerCount();
        workspace.wrapConvState =
                TornadoWorkspaces.zeroedFloats(recurrent * config.convStateSize());
        workspace.wrapDeltaState =
                TornadoWorkspaces.zeroedFloats(recurrent * config.deltaNetStateSize());

        // [0] = position, [1] = table-local KV slot.
        workspace.positionHolder = TornadoWorkspaces.ints(2);
        workspace.temp = TornadoWorkspaces.floats(1 + ((config.dim() + localSize - 1) / localSize));
        workspace.tempFFN =
                TornadoWorkspaces.floats(1 + ((config.dim() + localSize - 1) / localSize));
        workspace.tempLogits =
                TornadoWorkspaces.floats(1 + ((config.dim() + localSize - 1) / localSize));

        allocateBatchWorkspace(config, kvDim);

        // Sized by the blocks that attend, not by the block count: see keyValueLayerIndex.
        // This family has FP16 key/value kernels in all three execution modes, so it says yes and
        // lets the storage options decide.
        fillKvFields(fields, config, kvDim, config.keyValueLayerCount(), true);
    }

    // @formatter:off
    /**
     * The chunk-wide scratch batched prefill needs, when a batch width was configured.
     *
     * <p>Allocated here rather than beside the generic batch buffers in {@code State} because they
     * are this family's: a fused query/gate projection twice a query's width, a convolved {@code q
     * ‖ k ‖ v} of unequal parts, and the delta-net's own inputs and readout. The generic ones are
     * sized from {@code batchQDim}/{@code batchKvDim}, which cannot describe these.
     *
     * <p>The width comes from the same place {@code State}'s own batch buffers take it — how this
     * state was built — and not from the execution policy, which is resolved per generation and
     * would report a width this workspace was never sized for.
     *
     * <p>The recurrent <b>state</b> is not among them. It is one allocation for the whole session,
     * updated in place by both the batched and the single-token graphs — allocating a second copy
     * for prefill is what would break the continuity the mode exists to preserve.
     */
    // @formatter:on
    private void allocateBatchWorkspace(Qwen35Configuration config, int kvDim) {
        int batch = prefillBatchWidth;
        if (batch <= 1) {
            return;
        }
        workspace.wrapNormedBatch = TornadoWorkspaces.floats(batch * config.dim());
        // The same chunk in FP16, for the tensor-core projections. The MMA row tile is sixteen, so
        // the buffer carries whole tiles even when the chunk does not fill the last one.
        workspace.wrapNormedFP16Batch =
                TornadoWorkspaces.halfFloats(((batch + 15) / 16) * 16 * config.dim());
        // Gate and up land here before SwiGLU combines them: the tensor-core path cannot fuse the
        // combine, because that would mean reading fragment elements.
        workspace.wrapGateBatch = TornadoWorkspaces.floats(batch * config.hiddenDim());
        workspace.wrapUpBatch = TornadoWorkspaces.floats(batch * config.hiddenDim());
        workspace.wrapHbFP16BatchMMA =
                TornadoWorkspaces.halfFloats(((batch + 15) / 16) * 16 * config.hiddenDim());
        workspace.wrapFFNDownBatch = TornadoWorkspaces.floats(batch * config.dim());
        workspace.wrapSsmOutFP16Batch =
                TornadoWorkspaces.halfFloats(((batch + 15) / 16) * 16 * config.deltaNetValueDim());
        workspace.wrapQGateBatch = TornadoWorkspaces.floats(batch * config.queryGateDim());
        workspace.wrapAttnQBatch =
                TornadoWorkspaces.floats(batch * config.attentionOutputInputDim());
        workspace.wrapAttnGateBatch =
                TornadoWorkspaces.floats(batch * config.attentionOutputInputDim());
        workspace.wrapSsmQkvBatch = TornadoWorkspaces.floats(batch * config.deltaNetConvDim());
        workspace.wrapSsmConvOutBatch = TornadoWorkspaces.floats(batch * config.deltaNetConvDim());
        workspace.wrapSsmZBatch = TornadoWorkspaces.floats(batch * config.deltaNetValueDim());
        workspace.wrapSsmAlphaBatch = TornadoWorkspaces.floats(batch * config.numberOfValueHeads());
        workspace.wrapSsmBetaBatch = TornadoWorkspaces.floats(batch * config.numberOfValueHeads());
        workspace.wrapSsmQBatch = TornadoWorkspaces.floats(batch * config.deltaNetKeyDim());
        workspace.wrapSsmKBatch = TornadoWorkspaces.floats(batch * config.deltaNetKeyDim());
        workspace.wrapSsmVBatch = TornadoWorkspaces.floats(batch * config.deltaNetValueDim());
        workspace.wrapSsmOutBatch = TornadoWorkspaces.floats(batch * config.deltaNetValueDim());
        // The attention scores, for the FP16 key/value kernel that computes each dot product once.
        // Sized to the context capacity because a row's causal range can reach any position in
        // it; a span per (row, head) so every workgroup of a launch writes and reads its own.
        // Never uploaded, downloaded or reset: a launch reads only what it wrote.
        // The dequantize-then-GEMM scratch, only at the widths whose GEMM tiles the chunk fills:
        // one matrix, the largest projection that takes the pair (gate/up and the Q4_1 ffn_down:
        // hiddenDim x dim; the Q5_K ssm_out, dim x valueDim, is smaller), reused in turn.
        if (Qwen35Configuration.dequantGemmWidth(batch)) {
            workspace.wrapDequantScratchFP16 =
                    TornadoWorkspaces.halfFloats(
                            Math.toIntExact((long) config.hiddenDim() * config.dim()));
        }
        if (storageOptions().usesFp16KeyValueCache()) {
            // The capacity rounded up to whole 32-key tiles: the tensor-core kernels' transposed
            // regions are padded to them; the other kernels use the first contextLength of each
            // (row, head) span and never read past it.
            workspace.wrapAttnScoresBatch =
                    TornadoWorkspaces.floats(
                            Math.toIntExact(
                                    (long) batch
                                            * config.numberOfHeads()
                                            * Qwen35Configuration.attentionScoreKeys(
                                                    config.contextLength())));
            // The tensor-core attention's staging, at the widths its query tiles divide; the
            // kernel is dispatched from the same answer (Qwen35Configuration.attentionStageHalves).
            long stageHalves =
                    Qwen35Configuration.attentionStageHalves(batch, config.numberOfHeads());
            if (stageHalves > 0) {
                workspace.wrapAttnStageFP16 =
                        TornadoWorkspaces.halfFloats(Math.toIntExact(stageHalves));
            }
        }
    }
}
