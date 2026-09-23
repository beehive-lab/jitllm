package org.beehive.jllm.inference.state;

import java.util.stream.Stream;
import org.beehive.jllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jllm.model.Configuration;
import org.beehive.jllm.model.phi3.Phi3Configuration;
import org.beehive.jllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jllm.tensor.standard.FloatTensor;

public class Phi3State extends State {
    // Phi3-specific fields for QKV processing
    public final FloatTensor
            qkv; // Combined QKV buffer: op_size = dim + 2 * (n_kv_heads * head_dim)

    // Phi3-specific fields for FFN gate/up processing
    public final FloatTensor hbG; // Gate states buffer
    public final FloatTensor hbU; // Up states buffer

    public Phi3State(Configuration config, int batchsize) {
        this(config, batchsize, null);
    }

    /**
     * @param lease the KV lease whose shared storage this state addresses, or {@code null} to
     *     allocate its own arrays
     */
    public Phi3State(
            Configuration config, int batchsize, org.beehive.jllm.runtime.kv.KvLease lease) {
        super(config, batchsize, lease);

        // Initialize Phi3-specific fields
        Phi3Configuration phi3Config = (Phi3Configuration) config;

        // QKV buffer size: op_size = num_heads * head_dim + 2 * (num_key_value_heads * head_dim)
        int opSize =
                phi3Config.dim() + 2 * (phi3Config.numberOfKeyValueHeads() * phi3Config.headSize());
        this.qkv = ArrayFloatTensor.allocate(opSize);

        // FFN gate and up state buffers
        this.hbG = ArrayFloatTensor.allocate(phi3Config.hiddenDim());
        this.hbU = ArrayFloatTensor.allocate(phi3Config.hiddenDim());

        // TornadoVM wrappers for GPU acceleration
        this.workspace.wrapQkv = TornadoWorkspaces.floats(opSize);
        this.workspace.wrapHbG = TornadoWorkspaces.floats(phi3Config.hiddenDim());
        this.workspace.wrapHbU = TornadoWorkspaces.floats(phi3Config.hiddenDim());
    }

    @Override
    protected StateFields createStateFields(Configuration config) {
        StateFields fields = new StateFields();

        Phi3Configuration phi3Config = (Phi3Configuration) config;

        // Phi3-specific dimensions
        int dim = phi3Config.dim();
        int headSize = phi3Config.headSize();
        int nHeads = phi3Config.numberOfHeads();
        int nKvHeads = phi3Config.numberOfKeyValueHeads();
        int kvDim = (dim * nKvHeads) / nHeads;
        int hiddenDim = phi3Config.hiddenDim();
        int contextLength = phi3Config.contextLength();
        int vocabSize = phi3Config.vocabularySize();
        int nLayers = phi3Config.numberOfLayers();

        // Standard tensor allocations for Phi3
        fields.x = ArrayFloatTensor.allocate(dim);
        fields.xb = ArrayFloatTensor.allocate(dim); // Used for attention output
        fields.xb2 = ArrayFloatTensor.allocate(dim); // Used for residual connections
        fields.hb = ArrayFloatTensor.allocate(2 * hiddenDim); // Combined gate/up buffer
        fields.hb2 = ArrayFloatTensor.allocate(hiddenDim); // FFN output buffer

        // Attention-related tensors
        fields.q = ArrayFloatTensor.allocate(dim); // Query states
        fields.k = ArrayFloatTensor.allocate(kvDim); // Key states
        fields.v = ArrayFloatTensor.allocate(kvDim); // Value states
        fields.att = ArrayFloatTensor.allocate(nHeads, contextLength); // Attention scores

        // Output logits
        fields.logits = ArrayFloatTensor.allocate(vocabSize);

        // Key-value cache with Phi3 dimensions
        fields.keyCache =
                Stream.generate(() -> allocateKeyValue(contextLength, kvDim))
                        .limit(nLayers)
                        .toArray(FloatTensor[]::new);
        fields.valueCache =
                Stream.generate(() -> allocateKeyValue(contextLength, kvDim))
                        .limit(nLayers)
                        .toArray(FloatTensor[]::new);

        // TornadoVM wrapper arrays for GPU acceleration
        switch (config.quantization()) {
            case "FP16" -> TornadoWorkspaces.activationFP16(workspace, config.dim());
            case "Q8_0" -> TornadoWorkspaces.activationQ8_0(workspace, config.dim());
            default ->
                    throw new UnsupportedOperationException(
                            "Unsupported quantization format: " + config.quantization());
        }
        workspace.wrapX = TornadoWorkspaces.floats(dim);
        workspace.wrapXb = TornadoWorkspaces.floats(dim);
        workspace.wrapXFP16 = TornadoWorkspaces.halfFloats(dim);
        workspace.wrapXbFP16 = TornadoWorkspaces.halfFloats(dim);
        workspace.wrapXb2 = TornadoWorkspaces.floats(dim);
        workspace.wrapHb = TornadoWorkspaces.floats(2 * hiddenDim);
        workspace.wrapHb2 = TornadoWorkspaces.floats(hiddenDim);
        workspace.wrapLogits = TornadoWorkspaces.floats(vocabSize);
        workspace.wrapQ = TornadoWorkspaces.floats(dim);
        workspace.wrapK = TornadoWorkspaces.floats(kvDim);
        workspace.wrapV = TornadoWorkspaces.floats(kvDim);

        // KV cache wrappers
        // KV cache: leased from the manager's pool when this state holds a lease, otherwise
        // allocated here, block-major when paged and contiguous when not.
        fillKvFields(fields, config, kvDim, false);

        // Attention wrapper
        workspace.wrapAtt = TornadoWorkspaces.floats(nHeads * contextLength);

        // Position holder for GPU operations
        // [0] = position, [1] = table-local KV slot.
        workspace.positionHolder = TornadoWorkspaces.ints(2);

        // Temporary arrays for reductions and operations
        workspace.temp = TornadoWorkspaces.floats(1 + ((dim + localSize - 1) / localSize));
        workspace.tempFFN = TornadoWorkspaces.floats(1 + ((hiddenDim + localSize - 1) / localSize));
        workspace.tempLogits =
                TornadoWorkspaces.floats(1 + ((vocabSize + localSize - 1) / localSize));

        return fields;
    }
}
