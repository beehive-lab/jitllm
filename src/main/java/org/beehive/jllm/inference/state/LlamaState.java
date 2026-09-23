package org.beehive.jllm.inference.state;

import java.util.stream.Stream;
import org.beehive.jllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tensor.standard.FloatTensor;

/**
 * Represents the state of the Llama model during inference. This class extends {@link State} to
 * include model-specific functionalities and configurations tailored for the Llama model.
 *
 * <p><b>Note 1:</b> LlamaState contains additional fields for TornadoVM wrappers to enable
 * GPU-accelerated processing of the model.
 *
 * <p><b>Note 2:</b> This state implementation is also used for the Mistral model.
 */
public final class LlamaState extends State {

    public LlamaState(Configuration config, int batchsize) {
        this(config, batchsize, null);
    }

    /**
     * @param lease the KV lease whose shared storage this state addresses, or {@code null} to
     *     allocate its own arrays. A lease carrying backend storage is what makes several sessions
     *     share one pool instead of holding a copy each.
     */
    public LlamaState(
            Configuration config, int batchsize, org.beehive.jllm.runtime.kv.KvLease lease) {
        super(config, batchsize, lease);
    }

    @Override
    protected StateFields createStateFields(Configuration config) {
        StateFields fields = new StateFields();

        // Allocation with Llama/Mistral dimensions
        fields.x = ArrayFloatTensor.allocate(config.dim());
        fields.xb = ArrayFloatTensor.allocate(config.dim());
        fields.xb2 = ArrayFloatTensor.allocate(config.dim());
        fields.hb = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.q = ArrayFloatTensor.allocate(config.dim());
        fields.k = ArrayFloatTensor.allocate(config.dim());
        fields.v = ArrayFloatTensor.allocate(config.dim());
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());

        // Key-value cache with Llama/Mistral dimensions
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        fields.keyCache =
                Stream.generate(() -> allocateKeyValue(config.contextLength(), kvDim))
                        .limit(config.numberOfLayers())
                        .toArray(FloatTensor[]::new);
        fields.valueCache =
                Stream.generate(() -> allocateKeyValue(config.contextLength(), kvDim))
                        .limit(config.numberOfLayers())
                        .toArray(FloatTensor[]::new);

        // TornadoVM wrappers with Llama/Mistral dimensions
        workspace.wrapX = TornadoWorkspaces.floats(config.dim());
        workspace.wrapXb = TornadoWorkspaces.floats(config.dim());
        workspace.wrapXb2 = TornadoWorkspaces.floats(config.dim());
        workspace.wrapHb = TornadoWorkspaces.floats(config.hiddenDim());
        workspace.wrapHb2 = TornadoWorkspaces.floats(config.hiddenDim());

        switch (config.quantization()) {
            case "FP16" -> TornadoWorkspaces.activationFP16(workspace, config.dim());
            case "Q8_0" -> TornadoWorkspaces.activationQ8_0(workspace, config.dim());
            default ->
                    throw new UnsupportedOperationException(
                            "Unsupported quantization format: " + config.quantization());
        }
        workspace.wrapLogits = TornadoWorkspaces.floats(config.vocabularySize());
        workspace.wrapQ = TornadoWorkspaces.floats(config.dim());
        workspace.wrapK = TornadoWorkspaces.floats(config.dim());
        workspace.wrapV = TornadoWorkspaces.floats(config.dim());

        workspace.wrapXFP16 = TornadoWorkspaces.halfFloats(config.dim());
        workspace.wrapXbFP16 = TornadoWorkspaces.halfFloats(config.dim());
        // KV cache: leased from the manager's pool when this state holds a lease, otherwise
        // allocated here, block-major when paged and contiguous when not.
        fillKvFields(fields, config, kvDim, true);
        workspace.wrapAtt =
                TornadoWorkspaces.floats(config.numberOfHeads() * config.contextLength());
        workspace.wrapAttSplit =
                TornadoWorkspaces.floats(
                        config.numberOfHeads() * SPLIT_KV * (config.headSize() + 2));
        // [0] = position, [1] = table-local KV slot (always 0 while the table is this state's own).
        workspace.positionHolder = TornadoWorkspaces.ints(2);

        // Temporary arrays
        workspace.temp = TornadoWorkspaces.floats(1 + ((config.dim() + localSize - 1) / localSize));
        workspace.tempFFN =
                TornadoWorkspaces.floats(1 + ((config.dim() + localSize - 1) / localSize));
        workspace.tempLogits =
                TornadoWorkspaces.floats(1 + ((config.dim() + localSize - 1) / localSize));

        return fields;
    }
}
