package org.beehive.jitllm.inference.state;

import java.util.stream.Stream;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.devstral.DevstralConfiguration;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * State for Devstral 2 models where head_dim != dim/num_heads. Allocates Q with qDim (num_heads *
 * head_dim) and K/V with kvDim (num_kv_heads * head_dim).
 */
public final class DevstralState extends State {

    public DevstralState(Configuration config, int batchsize) {
        this(config, batchsize, null);
    }

    /**
     * @param lease the KV lease whose shared storage this state addresses, or {@code null} to
     *     allocate its own arrays
     */
    public DevstralState(
            Configuration config, int batchsize, org.beehive.jitllm.runtime.kv.KvLease lease) {
        super(config, batchsize, lease);
    }

    @Override
    protected StateFields createStateFields(Configuration config) {
        DevstralConfiguration dc = (DevstralConfiguration) config;
        StateFields fields = new StateFields();

        int qDim = dc.qDim();
        int kvDim = dc.kvDim();

        fields.x = ArrayFloatTensor.allocate(dc.dim());
        fields.xb = ArrayFloatTensor.allocate(dc.dim());
        fields.xb2 = ArrayFloatTensor.allocate(dc.dim());
        fields.hb = ArrayFloatTensor.allocate(dc.hiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(dc.hiddenDim());
        fields.q = ArrayFloatTensor.allocate(qDim);
        fields.k = ArrayFloatTensor.allocate(kvDim);
        fields.v = ArrayFloatTensor.allocate(kvDim);
        fields.att = ArrayFloatTensor.allocate(dc.numberOfHeads(), dc.contextLength());
        fields.logits = ArrayFloatTensor.allocate(dc.vocabularySize());

        fields.keyCache =
                Stream.generate(() -> allocateKeyValue(dc.contextLength(), kvDim))
                        .limit(dc.numberOfLayers())
                        .toArray(FloatTensor[]::new);
        fields.valueCache =
                Stream.generate(() -> allocateKeyValue(dc.contextLength(), kvDim))
                        .limit(dc.numberOfLayers())
                        .toArray(FloatTensor[]::new);

        // TornadoVM wrappers
        workspace.wrapX = TornadoWorkspaces.floats(dc.dim());
        workspace.wrapXb = TornadoWorkspaces.floats(dc.dim());
        workspace.wrapXb2 = TornadoWorkspaces.floats(dc.dim());
        workspace.wrapHb = TornadoWorkspaces.floats(dc.hiddenDim());
        workspace.wrapHb2 = TornadoWorkspaces.floats(dc.hiddenDim());

        switch (dc.quantization()) {
            case "FP16" -> TornadoWorkspaces.activationFP16(workspace, dc.dim());
            case "Q8_0" -> TornadoWorkspaces.activationQ8_0(workspace, dc.dim());
            default ->
                    throw new UnsupportedOperationException(
                            "Unsupported quantization format: " + dc.quantization());
        }
        workspace.wrapLogits = TornadoWorkspaces.floats(dc.vocabularySize());
        workspace.wrapQ = TornadoWorkspaces.floats(qDim);
        workspace.wrapK = TornadoWorkspaces.floats(kvDim);
        workspace.wrapV = TornadoWorkspaces.floats(kvDim);

        // The packed-integer projections' quantized activation, sized for the widest one they
        // read: the FFN down projection's hidden width, or the model width (qDim is narrower).
        int widest = Math.max(Math.max(dc.dim(), dc.hiddenDim()), qDim);
        TornadoWorkspaces.packedActivation(workspace, widest);

        workspace.wrapXFP16 = TornadoWorkspaces.halfFloats(dc.dim());
        workspace.wrapXbFP16 = TornadoWorkspaces.halfFloats(dc.dim());
        // KV cache: leased from the manager's pool when this state holds a lease, otherwise
        // allocated here, block-major when paged and contiguous when not.
        // Half precision for the quantized weights, whose decode layers and batched prefill read
        // and write the FP16 cache; the F16-weight layers keep the FP32 cache they always had.
        fillKvFields(fields, dc, kvDim, !"FP16".equals(dc.quantization()));
        workspace.wrapAtt = TornadoWorkspaces.floats(dc.numberOfHeads() * dc.contextLength());
        // Split-KV decode attention's partial outputs: per head and partition, the head's
        // weighted values plus its running max and sum.
        workspace.wrapAttSplit =
                TornadoWorkspaces.floats(dc.numberOfHeads() * SPLIT_KV * (dc.headSize() + 2));
        // [0] = position, [1] = table-local KV slot.
        workspace.positionHolder = TornadoWorkspaces.ints(2);

        workspace.temp = TornadoWorkspaces.floats(1 + ((dc.dim() + localSize - 1) / localSize));
        workspace.tempFFN = TornadoWorkspaces.floats(1 + ((dc.dim() + localSize - 1) / localSize));
        workspace.tempLogits =
                TornadoWorkspaces.floats(1 + ((dc.dim() + localSize - 1) / localSize));

        return fields;
    }
}
