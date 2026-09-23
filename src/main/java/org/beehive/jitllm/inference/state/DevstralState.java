package org.beehive.jllm.inference.state;

import java.util.stream.Stream;
import org.beehive.jllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.devstral.DevstralConfiguration;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tensor.standard.FloatTensor;

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
            Configuration config, int batchsize, org.beehive.jllm.runtime.kv.KvLease lease) {
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

        workspace.wrapXFP16 = TornadoWorkspaces.halfFloats(dc.dim());
        workspace.wrapXbFP16 = TornadoWorkspaces.halfFloats(dc.dim());
        // KV cache: leased from the manager's pool when this state holds a lease, otherwise
        // allocated here, block-major when paged and contiguous when not.
        fillKvFields(fields, dc, kvDim, false);
        workspace.wrapAtt = TornadoWorkspaces.floats(dc.numberOfHeads() * dc.contextLength());
        // [0] = position, [1] = table-local KV slot.
        workspace.positionHolder = TornadoWorkspaces.ints(2);

        workspace.temp = TornadoWorkspaces.floats(1 + ((dc.dim() + localSize - 1) / localSize));
        workspace.tempFFN = TornadoWorkspaces.floats(1 + ((dc.dim() + localSize - 1) / localSize));
        workspace.tempLogits =
                TornadoWorkspaces.floats(1 + ((dc.dim() + localSize - 1) / localSize));

        return fields;
    }
}
