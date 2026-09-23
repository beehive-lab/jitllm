package org.beehive.jitllm.inference.state;

import java.util.stream.Stream;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.qwen2.Qwen2MoEConfiguration;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;

public class Qwen2MoEState extends Qwen2State {

    // Router output scores, one per expert (length = numberOfExperts).
    // Used to pick the top-k experts for the current token.
    public final FloatTensor routerLogits;

    // Scratch buffers for a single routed expert's internal FFN (length = moeHiddenDim).
    // hbE holds gate_proj(xb), hbE2 holds up_proj(xb); combined via silu(hbE) * hbE2.
    public final FloatTensor hbE;
    public final FloatTensor hbE2;

    // Scratch buffers for the shared expert's internal FFN (length = sharedExpertHiddenDim).
    // Same role as hbE/hbE2, but for the always-on shared expert instead of a routed one.
    public final FloatTensor hbS;
    public final FloatTensor hbS2;

    // Temporary holder for a single expert's down-projected output (length = dim),
    // before it is weighted and accumulated into the residual stream (state.x).
    public final FloatTensor yTmp;

    /**
     * The router's selected expert identifiers and their routing weights — <b>fixed workspace</b>,
     * {@code numberOfExpertsUsed()} long.
     */
    public final int[] selectedExperts;

    public final float[] selectedExpertWeights;

    // TornadoVM buffers for the single-token GPU MoE path.  These are deliberately
    // separate from the CPU FloatTensor fields above: TaskGraph kernels operate on
    // TornadoVM arrays that can remain resident on the device between tasks.

    // TornadoVM buffers for the batch-prefill MoE path.
    // Their shapes use the configured maximum batch size so TaskGraphs stay fixed.

    public Qwen2MoEState(Configuration config, int batchsize) {
        this(config, batchsize, null);
    }

    /**
     * @param lease the KV lease whose shared storage this state addresses, or {@code null} to
     *     allocate its own arrays
     */
    public Qwen2MoEState(
            Configuration config, int batchsize, org.beehive.jitllm.runtime.kv.KvLease lease) {
        super(config, batchsize, lease);
        Qwen2MoEConfiguration c = (Qwen2MoEConfiguration) config;
        this.routerLogits = ArrayFloatTensor.allocate(c.numberOfExperts());
        this.hbE = ArrayFloatTensor.allocate(c.moeHiddenDim());
        this.hbE2 = ArrayFloatTensor.allocate(c.moeHiddenDim());
        this.hbS = ArrayFloatTensor.allocate(c.sharedExpertHiddenDim());
        this.hbS2 = ArrayFloatTensor.allocate(c.sharedExpertHiddenDim());
        this.yTmp = ArrayFloatTensor.allocate(c.dim());
        this.selectedExperts = new int[c.numberOfExpertsUsed()];
        this.selectedExpertWeights = new float[c.numberOfExpertsUsed()];

        this.workspace.wrapRouterLogits = TornadoWorkspaces.floats(c.numberOfExperts());
        this.workspace.wrapSelectedExperts = TornadoWorkspaces.ints(c.numberOfExpertsUsed());
        this.workspace.wrapRoutingWeights = TornadoWorkspaces.floats(c.numberOfExpertsUsed());
        this.workspace.wrapExpertGate =
                TornadoWorkspaces.floats(c.moeHiddenDim() * c.numberOfExpertsUsed());
        this.workspace.wrapSharedGate = TornadoWorkspaces.floats(c.sharedExpertHiddenDim());
        this.workspace.wrapSharedOutput = TornadoWorkspaces.floats(c.dim());

        int gpuBatchSize = Integer.getInteger("jitllm.prefillBatchSize", 1);
        if (gpuBatchSize > 1) {
            int assignments = gpuBatchSize * c.numberOfExpertsUsed();
            this.workspace.wrapRouterLogitsBatch =
                    TornadoWorkspaces.floats(gpuBatchSize * c.numberOfExperts());
            this.workspace.activeBatchSizeHolder = TornadoWorkspaces.ints(1);
            TornadoWorkspaces.activeBatchSize(this.workspace, gpuBatchSize);
            this.workspace.wrapSelectedExpertsBatch = TornadoWorkspaces.ints(assignments);
            this.workspace.wrapRoutingWeightsBatch = TornadoWorkspaces.floats(assignments);
            this.workspace.wrapGroupedAssignmentIds = TornadoWorkspaces.ints(assignments);
            this.workspace.wrapGroupedPositionByAssignment = TornadoWorkspaces.ints(assignments);
            this.workspace.wrapGroupedExpertHidden =
                    TornadoWorkspaces.floats(assignments * c.moeHiddenDim());
            this.workspace.wrapGroupedExpertDown = TornadoWorkspaces.floats(assignments * c.dim());
            this.workspace.wrapSharedHiddenBatch =
                    TornadoWorkspaces.floats(gpuBatchSize * c.sharedExpertHiddenDim());
            this.workspace.wrapSharedWeightBatch = TornadoWorkspaces.floats(gpuBatchSize);
        } else {
            this.workspace.wrapRouterLogitsBatch = null;
            this.workspace.activeBatchSizeHolder = null;
            this.workspace.wrapSelectedExpertsBatch = null;
            this.workspace.wrapRoutingWeightsBatch = null;
            this.workspace.wrapGroupedAssignmentIds = null;
            this.workspace.wrapGroupedPositionByAssignment = null;
            this.workspace.wrapGroupedExpertHidden = null;
            this.workspace.wrapGroupedExpertDown = null;
            this.workspace.wrapSharedHiddenBatch = null;
            this.workspace.wrapSharedWeightBatch = null;
        }
    }

    @Override
    protected StateFields createStateFields(Configuration configuration) {
        StateFields fields = new StateFields();

        Qwen2MoEConfiguration config = (Qwen2MoEConfiguration) configuration;

        int nEmbdGqa = config.kvDim();

        fields.x = ArrayFloatTensor.allocate(config.dim());
        fields.xb = ArrayFloatTensor.allocate(config.dim());
        fields.xb2 = ArrayFloatTensor.allocate(config.dim());
        fields.hb = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.q = ArrayFloatTensor.allocate(config.dim());
        fields.k = ArrayFloatTensor.allocate(config.kvDim());
        fields.v = ArrayFloatTensor.allocate(config.kvDim());
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());

        fields.keyCache =
                Stream.generate(() -> allocateKeyValue(config.contextLength(), nEmbdGqa))
                        .limit(config.numberOfLayers())
                        .toArray(FloatTensor[]::new);
        fields.valueCache =
                Stream.generate(() -> allocateKeyValue(config.contextLength(), nEmbdGqa))
                        .limit(config.numberOfLayers())
                        .toArray(FloatTensor[]::new);

        switch (config.quantization()) {
            case "FP16" -> TornadoWorkspaces.activationFP16(workspace, config.dim());
            case "Q8_0" -> TornadoWorkspaces.activationQ8_0(workspace, config.dim());
            default ->
                    throw new UnsupportedOperationException(
                            "Unsupported quantization format: " + config.quantization());
        }
        workspace.wrapX = TornadoWorkspaces.floats(config.dim());
        workspace.wrapXb = TornadoWorkspaces.floats(config.dim());
        workspace.wrapXbFP16 = TornadoWorkspaces.halfFloats(config.dim());
        workspace.wrapXb2 = TornadoWorkspaces.floats(config.dim());
        workspace.wrapHb = TornadoWorkspaces.floats(config.hiddenDim());
        workspace.wrapHb2 = TornadoWorkspaces.floats(config.hiddenDim());

        workspace.wrapLogits = TornadoWorkspaces.floats(config.vocabularySize());
        workspace.wrapQ = TornadoWorkspaces.floats(config.dim());
        workspace.wrapK = TornadoWorkspaces.floats(config.kvDim());
        workspace.wrapV = TornadoWorkspaces.floats(config.kvDim());

        // KV cache: leased from the manager's pool when this state holds a lease, otherwise
        // allocated here, block-major when paged and contiguous when not.
        fillKvFields(fields, config, nEmbdGqa, false);
        workspace.wrapAtt =
                TornadoWorkspaces.floats(config.numberOfHeads() * config.contextLength());
        // [0] = position, [1] = table-local KV slot.
        workspace.positionHolder = TornadoWorkspaces.ints(2);

        // State invokes this override before the Qwen2State constructor body runs,
        // so use the Qwen2 work-group size directly instead of State.localSize.
        workspace.temp =
                TornadoWorkspaces.floats(
                        1 + ((config.dim() + QWEN2_LOCAL_SIZE - 1) / QWEN2_LOCAL_SIZE));
        workspace.tempFFN =
                TornadoWorkspaces.floats(
                        1 + ((config.dim() + QWEN2_LOCAL_SIZE - 1) / QWEN2_LOCAL_SIZE));
        workspace.tempLogits =
                TornadoWorkspaces.floats(
                        1 + ((config.dim() + QWEN2_LOCAL_SIZE - 1) / QWEN2_LOCAL_SIZE));

        return fields;
    }
}
