package org.beehive.jitllm.inference.state;

import java.util.stream.Stream;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.qwen3.Qwen3Configuration;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;

/**
 * Represents the state of the Qwen3 model during inference. This class extends {@link State} to
 * include model-specific functionalities and configurations tailored for the Qwen3 model.
 *
 * <p><b>Note 1:</b> Qwen3State contains additional fields for TornadoVM wrappers to enable
 * GPU-accelerated processing of the model.
 */
public final class Qwen3State extends State {

    // Qwen3 specific fields
    // Temporary buffers for intermediate calculations.
    /**
     * Q/K RMS-norm scratch, per head. Qwen3 normalises Q and K before RoPE; no other family does.
     */
    public Qwen3State(Configuration config, int batchsize) {
        this(config, batchsize, null);
    }

    /**
     * @param lease the KV lease whose shared storage this state addresses, or {@code null} to
     *     allocate its own arrays
     */
    public Qwen3State(
            Configuration config, int batchsize, org.beehive.jitllm.runtime.kv.KvLease lease) {
        super(config, batchsize, lease);
        // Initialize Qwen3-specific fields
        Qwen3Configuration qwen3config = (Qwen3Configuration) config;
        int nEmbdHead = qwen3config.numberOfHeads();
        this.workspace.tempQcur = TornadoWorkspaces.floats(nEmbdHead);
        this.workspace.tempKcur = TornadoWorkspaces.floats(nEmbdHead);
    }

    @Override
    protected int batchQDim(Configuration config) {
        Qwen3Configuration q3 = (Qwen3Configuration) config;
        return q3.numberOfHeadsKey() * q3.numberOfHeads();
    }

    @Override
    protected int batchKvDim(Configuration config) {
        Qwen3Configuration q3 = (Qwen3Configuration) config;
        return q3.numberOfHeadsValue() * q3.numberOfKeyValueHeads();
    }

    @Override
    protected StateFields createStateFields(Configuration configuration) {
        StateFields fields = new StateFields();

        Qwen3Configuration config = (Qwen3Configuration) configuration;

        // Qwen3-specific sizes
        int nHeadKv = config.numberOfKeyValueHeads();
        int nEmbdHeadK = config.numberOfHeadsKey();
        int nEmbdKGqa = nEmbdHeadK * nHeadKv;
        int nEmbdHeadV = config.numberOfHeadsValue();
        int nEmbdVGqa = nEmbdHeadV * nHeadKv;
        int nEmbdGqa = nEmbdVGqa;

        // Qwen3-specific allocation logic
        fields.x = ArrayFloatTensor.allocate(config.dim());
        fields.xb = ArrayFloatTensor.allocate(nEmbdHeadK * config.numberOfHeads());
        fields.xb2 = ArrayFloatTensor.allocate(config.dim());
        fields.hb = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.q = ArrayFloatTensor.allocate(nEmbdHeadK * config.numberOfHeads());
        fields.k = ArrayFloatTensor.allocate(nEmbdKGqa);
        fields.v = ArrayFloatTensor.allocate(nEmbdKGqa);
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());

        // Key-value cache with Qwen3 dimensions
        fields.keyCache =
                Stream.generate(() -> allocateKeyValue(config.contextLength(), nEmbdGqa))
                        .limit(config.numberOfLayers())
                        .toArray(FloatTensor[]::new);
        fields.valueCache =
                Stream.generate(() -> allocateKeyValue(config.contextLength(), nEmbdGqa))
                        .limit(config.numberOfLayers())
                        .toArray(FloatTensor[]::new);

        // TornadoVM wrappers with Qwen3-specific sizes

        switch (config.quantization()) {
            case "FP16" -> TornadoWorkspaces.activationFP16(workspace, config.dim());
            case "Q8_0" -> TornadoWorkspaces.activationQ8_0(workspace, config.dim());
            default ->
                    throw new UnsupportedOperationException(
                            "Unsupported quantization format: " + config.quantization());
        }

        workspace.wrapX = TornadoWorkspaces.floats(config.dim());
        workspace.wrapXb = TornadoWorkspaces.floats(nEmbdHeadK * config.numberOfHeads());
        workspace.wrapXbFP16 = TornadoWorkspaces.halfFloats(nEmbdHeadK * config.numberOfHeads());

        workspace.wrapXb2 = TornadoWorkspaces.floats(config.dim());
        workspace.wrapHb = TornadoWorkspaces.floats(config.hiddenDim());
        workspace.wrapHb2 = TornadoWorkspaces.floats(config.hiddenDim());
        workspace.wrapLogits = TornadoWorkspaces.floats(config.vocabularySize());
        workspace.wrapQ = TornadoWorkspaces.floats(nEmbdHeadK * config.numberOfHeads());
        workspace.wrapK = TornadoWorkspaces.floats(nEmbdKGqa);
        workspace.wrapV = TornadoWorkspaces.floats(nEmbdKGqa);
        // KV cache: leased from the manager's pool when this state holds a lease, otherwise
        // allocated here, block-major when paged and contiguous when not.
        fillKvFields(fields, config, nEmbdGqa, true);
        workspace.wrapAtt =
                TornadoWorkspaces.floats(config.numberOfHeads() * config.contextLength());
        // Qwen3 sizes the split-KV scratch by its value-head dimension, not by headSize.
        workspace.wrapAttSplit =
                TornadoWorkspaces.floats(
                        config.numberOfHeads() * SPLIT_KV * (config.numberOfHeadsValue() + 2));
        // [0] = position, [1] = table-local KV slot.
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
