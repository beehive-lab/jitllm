package org.beehive.jitllm.inference.state;

import java.util.stream.Stream;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.gptoss.GptOssConfiguration;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * The {@code gpt-oss} state. The query and the attention output are {@code heads * headSize} wide,
 * which is wider than the residual stream, so they get their own buffers.
 */
public class GptOssState extends State {

    static final int GPT_OSS_LOCAL_SIZE = 32;

    /** The attention output, before the output projection ({@code queryDim}). */
    public final FloatTensor attentionOut;

    /** Router scores, one per expert. */
    public final FloatTensor routerLogits;

    /** One expert's gate and up projections ({@code expertHiddenDim}). */
    public final FloatTensor expertHidden;

    public final FloatTensor expertHiddenUp;

    /** One expert's down projection ({@code dim}). */
    public final FloatTensor expertOut;

    public final int[] selectedExperts;

    public final float[] selectedExpertWeights;

    public GptOssState(Configuration config, int batchsize) {
        this(config, batchsize, null);
    }

    public GptOssState(
            Configuration config, int batchsize, org.beehive.jitllm.runtime.kv.KvLease lease) {
        super(config, batchsize, lease);
        GptOssConfiguration c = (GptOssConfiguration) config;
        // The logits graph's RMS reduction runs one work-group of this many threads.
        this.localSize = 256;
        TornadoWorkspaces.packedActivation(workspace, Math.max(c.dim(), c.queryDim()));
        this.attentionOut = ArrayFloatTensor.allocate(c.queryDim());
        this.routerLogits = ArrayFloatTensor.allocate(c.numberOfExperts());
        this.expertHidden = ArrayFloatTensor.allocate(c.expertHiddenDim());
        this.expertHiddenUp = ArrayFloatTensor.allocate(c.expertHiddenDim());
        this.expertOut = ArrayFloatTensor.allocate(c.dim());
        this.selectedExperts = new int[c.numberOfExpertsUsed()];
        this.selectedExpertWeights = new float[c.numberOfExpertsUsed()];

        workspace.wrapRouterLogits = TornadoWorkspaces.floats(c.numberOfExperts());
        workspace.wrapSelectedExperts = TornadoWorkspaces.ints(c.numberOfExpertsUsed());
        workspace.wrapRoutingWeights = TornadoWorkspaces.floats(c.numberOfExpertsUsed());
        workspace.wrapExpertGate =
                TornadoWorkspaces.floats(c.expertHiddenDim() * c.numberOfExpertsUsed());
        workspace.wrapExpertDown = TornadoWorkspaces.floats(c.dim() * c.numberOfExpertsUsed());
        workspace.wrapAttSplit =
                TornadoWorkspaces.floats(
                        c.numberOfHeads()
                                * org.beehive.jitllm.backend.tornado.kernels.GptOssKernels.SPLITS
                                * org.beehive.jitllm.backend.tornado.kernels.GptOssKernels.PARTIAL);
        if (prefillBatchWidth > 1) {
            allocateBatch(c, prefillBatchWidth);
        }
    }

    /** The batched prefill's chunk buffers: norms, int8 activations, routing and expert scratch. */
    private void allocateBatch(GptOssConfiguration c, int batch) {
        int assignments = batch * c.numberOfExpertsUsed();
        int hidden = c.expertHiddenDim();
        int widest = Math.max(c.queryDim(), Math.max(c.dim(), hidden));
        workspace.wrapNormedBatch = TornadoWorkspaces.floats(batch * c.dim());
        workspace.wrapQ8ActBatch = TornadoWorkspaces.bytes(batch * widest);
        workspace.wrapQ8ActScales = TornadoWorkspaces.floats(batch * widest / 32);
        workspace.wrapMoeLogitsBatch = TornadoWorkspaces.floats(batch * c.numberOfExperts());
        workspace.wrapMoeIdsBatch = TornadoWorkspaces.ints(assignments);
        workspace.wrapMoeWeightsBatch = TornadoWorkspaces.floats(assignments);
        workspace.wrapMoeSortedToken = TornadoWorkspaces.ints(assignments);
        workspace.wrapMoePosition = TornadoWorkspaces.ints(assignments);
        workspace.wrapMoeTiles =
                TornadoWorkspaces.ints(
                        1
                                + 3
                                        * org.beehive.jitllm.backend.tornado.kernels
                                                .Qwen35MoeBatchKernels.maxTiles(
                                                assignments, c.numberOfExperts()));
        workspace.wrapMoeHiddenBatch = TornadoWorkspaces.floats(assignments * hidden);
        workspace.wrapMoeHiddenQ8 = TornadoWorkspaces.bytes(assignments * hidden);
        workspace.wrapMoeHiddenScales = TornadoWorkspaces.floats(assignments * hidden / 32);
        workspace.wrapMoeOutBatch = TornadoWorkspaces.floats(assignments * c.dim());
    }

    @Override
    protected int batchQDim(Configuration config) {
        return config.queryDim();
    }

    @Override
    protected int batchKvDim(Configuration config) {
        return config.kvDim();
    }

    @Override
    protected StateFields createStateFields(Configuration configuration) {
        StateFields fields = new StateFields();
        GptOssConfiguration config = (GptOssConfiguration) configuration;
        int kvDim = config.kvDim();
        int qDim = config.queryDim();

        fields.x = ArrayFloatTensor.allocate(config.dim());
        fields.xb = ArrayFloatTensor.allocate(config.dim());
        fields.xb2 = ArrayFloatTensor.allocate(config.dim());
        fields.hb = ArrayFloatTensor.allocate(config.expertHiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(config.expertHiddenDim());
        fields.q = ArrayFloatTensor.allocate(qDim);
        fields.k = ArrayFloatTensor.allocate(kvDim);
        fields.v = ArrayFloatTensor.allocate(kvDim);
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());
        fields.keyCache =
                Stream.generate(() -> allocateKeyValue(config.contextLength(), kvDim))
                        .limit(config.numberOfLayers())
                        .toArray(FloatTensor[]::new);
        fields.valueCache =
                Stream.generate(() -> allocateKeyValue(config.contextLength(), kvDim))
                        .limit(config.numberOfLayers())
                        .toArray(FloatTensor[]::new);

        TornadoWorkspaces.activationQ8_0(workspace, config.dim());
        workspace.wrapX = TornadoWorkspaces.floats(config.dim());
        workspace.wrapXb = TornadoWorkspaces.floats(qDim);
        workspace.wrapXbFP16 = TornadoWorkspaces.halfFloats(qDim);
        workspace.wrapXb2 = TornadoWorkspaces.floats(config.dim());
        workspace.wrapHb = TornadoWorkspaces.floats(config.expertHiddenDim());
        workspace.wrapHb2 = TornadoWorkspaces.floats(config.expertHiddenDim());
        workspace.wrapLogits = TornadoWorkspaces.floats(config.vocabularySize());
        workspace.wrapQ = TornadoWorkspaces.floats(qDim);
        workspace.wrapK = TornadoWorkspaces.floats(kvDim);
        workspace.wrapV = TornadoWorkspaces.floats(kvDim);

        fillKvFields(fields, config, kvDim, true);
        workspace.wrapAtt =
                TornadoWorkspaces.floats(config.numberOfHeads() * config.contextLength());
        // [0] = position, [1] = table-local KV slot.
        workspace.positionHolder = TornadoWorkspaces.ints(2);

        int partials = 1 + (config.dim() + GPT_OSS_LOCAL_SIZE - 1) / GPT_OSS_LOCAL_SIZE;
        workspace.temp = TornadoWorkspaces.floats(partials);
        workspace.tempFFN = TornadoWorkspaces.floats(partials);
        workspace.tempLogits = TornadoWorkspaces.floats(partials);
        return fields;
    }
}
