package org.beehive.jllm.inference.state;

import java.util.stream.Stream;
import org.beehive.jllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.qwen2.Qwen2Configuration;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tensor.standard.FloatTensor;

public class Qwen2State extends State {

    protected static final int QWEN2_LOCAL_SIZE = 32;

    public Qwen2State(Configuration config, int batchsize) {
        this(config, batchsize, null);
    }

    /**
     * @param lease the KV lease whose shared storage this state addresses, or {@code null} to
     *     allocate its own arrays
     */
    public Qwen2State(
            Configuration config, int batchsize, org.beehive.jllm.runtime.kv.KvLease lease) {
        super(config, batchsize, lease);
        this.localSize = QWEN2_LOCAL_SIZE;
    }

    @Override
    protected StateFields createStateFields(Configuration configuration) {
        StateFields fields = new StateFields();

        Qwen2Configuration config = (Qwen2Configuration) configuration;

        int nEmbdGqa = config.kvDim();

        // with Qwen2-specific sizes
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

        // Key-value cache with Qwen2 dimensions
        fields.keyCache =
                Stream.generate(() -> allocateKeyValue(config.contextLength(), nEmbdGqa))
                        .limit(config.numberOfLayers())
                        .toArray(FloatTensor[]::new);
        fields.valueCache =
                Stream.generate(() -> allocateKeyValue(config.contextLength(), nEmbdGqa))
                        .limit(config.numberOfLayers())
                        .toArray(FloatTensor[]::new);

        // TornadoVM wrappers with Qwen2 dimensions
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
        fillKvFields(fields, config, nEmbdGqa, true);
        workspace.wrapAtt =
                TornadoWorkspaces.floats(config.numberOfHeads() * config.contextLength());
        // [0] = position, [1] = table-local KV slot.
        workspace.positionHolder = TornadoWorkspaces.ints(2);

        // Temporary arrays
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
