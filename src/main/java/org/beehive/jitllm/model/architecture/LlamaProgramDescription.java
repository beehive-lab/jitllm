package org.beehive.jitllm.model.architecture;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Set;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.llama.LlamaConfiguration;
import org.beehive.jitllm.program.BindingEntry;
import org.beehive.jitllm.program.BindingRole;
import org.beehive.jitllm.program.CapacityShape;
import org.beehive.jitllm.program.Direction;
import org.beehive.jitllm.program.InferenceProgram;
import org.beehive.jitllm.program.PhaseId;
import org.beehive.jitllm.program.PhaseSelection;
import org.beehive.jitllm.program.ProgramComponent;
import org.beehive.jitllm.program.ProgramSignature;
import org.beehive.jitllm.program.ResultId;
import org.beehive.jitllm.program.ValueId;
import org.beehive.jitllm.program.ValueType;
import org.beehive.jitllm.program.op.ArgMax;
import org.beehive.jitllm.program.op.Attention;
import org.beehive.jitllm.program.op.EmbeddingLookup;
import org.beehive.jitllm.program.op.KvAppend;
import org.beehive.jitllm.program.op.MatVec;
import org.beehive.jitllm.program.op.OperandRef;
import org.beehive.jitllm.program.op.ResidualAdd;
import org.beehive.jitllm.program.op.RmsNorm;
import org.beehive.jitllm.program.op.RoPE;
import org.beehive.jitllm.program.op.RopeLayout;
import org.beehive.jitllm.program.op.SwiGLU;
import org.beehive.jitllm.program.op.VocabProjection;
import org.beehive.jitllm.runtime.model.ArchitectureId;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.Shape;
import org.beehive.jitllm.runtime.tensor.TensorRole;

/**
 * Builds the {@code InferenceProgram} for Llama, single-token, at a given weight representation.
 *
 * <p>The description is backend-neutral even here: nothing in this class names a task graph, a
 * device or a fusion choice. It builds {@code program.**} types from a configuration and hands them
 * over.
 */
public final class LlamaProgramDescription {

    /** Names the description and the validator agree on. Not device buffers — operand names. */
    static final String X = "x";

    static final String XB = "xb";
    static final String XB2 = "xb2";
    static final String Q = "q";
    static final String K = "k";
    static final String V = "v";
    static final String HB = "hb";
    static final String HB2 = "hb2";
    static final String KEY_STORE = "keyCache";
    static final String VALUE_STORE = "valueCache";
    static final String LOGITS = "logits";
    static final String TOKENS = "tokens";
    static final String CONTROL = "control";
    static final String SAMPLED = "sampled";

    private static final Set<PhaseId> BOTH = EnumSet.allOf(PhaseId.class);
    private static final Set<PhaseId> DECODE = EnumSet.of(PhaseId.DECODE);

    private LlamaProgramDescription() {}

    /**
     * @param config the model's shape
     * @param weightDataType how the weights are represented <b>on the device</b> — F16 or Q8_0 for
     *     this backend. It parameterizes exactly the operations that read a weight, and nothing
     *     else: normalization, rotary, SwiGLU and the residuals accumulate in F32 whichever it is.
     *     Dequantization is a materialization concern fused inside {@code MatVec}, so a quantized
     *     representation adds <b>no operation and no component</b> — only these dtypes move
     * @param kvDataType how key/value storage is represented — <b>part of the signature</b>,
     *     because it changes the dtype of the fixed key/value bindings even though it leaves the
     *     component sequence identical
     * @param deviceSample whether sampling is device-resident; adds an {@link ArgMax} component
     * @param splitKvAttention whether split-KV attention is used; policy, not capability
     */
    public static InferenceProgram build(
            LlamaConfiguration config,
            DataType weightDataType,
            DataType kvDataType,
            boolean deviceSample,
            boolean splitKvAttention) {
        return build(
                ArchitectureId.of("llama"),
                config,
                weightDataType,
                kvDataType,
                deviceSample,
                splitKvAttention);
    }

    /**
     * The same sequence under another family's name.
     *
     * <p>The <b>architecture is still a parameter</b> rather than dropped: it is part of the
     * signature, so two families never share a compiled program by accident even when their shapes
     * coincide. What is shared here is the sequence; what is not shared is the identity.
     */
    public static InferenceProgram build(
            ArchitectureId architecture,
            Configuration config,
            DataType weightDataType,
            DataType kvDataType,
            boolean deviceSample,
            boolean splitKvAttention) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        int headSize = config.headSize();
        int layers = config.numberOfLayers();

        List<ProgramComponent> components = new ArrayList<>();
        components.add(
                new ProgramComponent.Leaf(
                        "embedding",
                        new EmbeddingLookup(
                                weight(TensorRole.TOKEN_EMBEDDING),
                                activation(TOKENS),
                                activation(X),
                                dim,
                                weightDataType),
                        BOTH));

        for (int layer = 0; layer < layers; layer++) {
            components.add(
                    layerComponent(
                            config, weightDataType, kvDataType, layer, dim, kvDim, headSize));
        }

        components.add(
                new ProgramComponent.Leaf(
                        "finalNorm",
                        new RmsNorm(
                                activation(X),
                                Optional.of(weight(TensorRole.OUTPUT_NORM)),
                                activation(X),
                                config.rmsNormEps(),
                                DataType.F32),
                        DECODE));
        components.add(
                new ProgramComponent.Leaf(
                        "vocabProjection",
                        new VocabProjection(
                                weight(TensorRole.OUTPUT),
                                activation(X),
                                activation(LOGITS),
                                config.vocabularySize(),
                                weightDataType),
                        DECODE));
        if (deviceSample) {
            components.add(
                    new ProgramComponent.Leaf(
                            "sample",
                            new ArgMax(activation(LOGITS), activation(SAMPLED), DataType.F32),
                            DECODE));
        }

        List<Integer> prefill = new ArrayList<>();
        List<Integer> decode = new ArrayList<>();
        for (int i = 0; i < components.size(); i++) {
            if (components.get(i).phases().contains(PhaseId.PREFILL)) {
                prefill.add(i);
            }
            decode.add(i);
        }

        ProgramSignature signature =
                new ProgramSignature(
                        architecture,
                        PolicyDescriptor.singleToken(deviceSample, splitKvAttention),
                        new CapacityShape(
                                config.contextLength(), layers, 1, config.contextLength(), 1, 1),
                        components,
                        List.of(
                                new PhaseSelection(PhaseId.PREFILL, prefill),
                                new PhaseSelection(PhaseId.DECODE, decode)),
                        bindings(config, weightDataType, kvDataType, deviceSample));
        return InferenceProgram.of(signature);
    }

    private static ProgramComponent layerComponent(
            Configuration config,
            DataType weightDataType,
            DataType kvDataType,
            int layer,
            int dim,
            int kvDim,
            int headSize) {
        String suffix = "_" + layer;
        List<ProgramComponent> inner =
                List.of(
                        leaf(
                                "attnNorm" + suffix,
                                new RmsNorm(
                                        activation(X),
                                        Optional.of(weight(TensorRole.ATTENTION_NORM)),
                                        activation(XB),
                                        config.rmsNormEps(),
                                        DataType.F32)),
                        leaf(
                                "q" + suffix,
                                new MatVec(
                                        weight(TensorRole.ATTENTION_QUERY),
                                        activation(XB),
                                        activation(Q),
                                        dim,
                                        dim,
                                        weightDataType)),
                        leaf(
                                "k" + suffix,
                                new MatVec(
                                        weight(TensorRole.ATTENTION_KEY),
                                        activation(XB),
                                        activation(K),
                                        kvDim,
                                        dim,
                                        weightDataType)),
                        leaf(
                                "v" + suffix,
                                new MatVec(
                                        weight(TensorRole.ATTENTION_VALUE),
                                        activation(XB),
                                        activation(V),
                                        kvDim,
                                        dim,
                                        weightDataType)),
                        leaf(
                                "rope" + suffix,
                                new RoPE(
                                        activation(Q),
                                        activation(K),
                                        Optional.of(weight(TensorRole.ROPE_FREQUENCIES)),
                                        headSize,
                                        config.ropeTheta(),
                                        RopeLayout.INTERLEAVED,
                                        DataType.F32)),
                        leaf(
                                "kvAppend" + suffix,
                                new KvAppend(
                                        activation(K),
                                        activation(V),
                                        activation(KEY_STORE),
                                        activation(VALUE_STORE),
                                        kvDim,
                                        kvDataType)),
                        leaf(
                                "attention" + suffix,
                                new Attention(
                                        activation(Q),
                                        activation(KEY_STORE),
                                        activation(VALUE_STORE),
                                        activation(XB),
                                        config.numberOfHeads(),
                                        config.numberOfKeyValueHeads(),
                                        headSize,
                                        (float) Math.sqrt(headSize),
                                        OptionalInt.empty(),
                                        kvDataType)),
                        leaf(
                                "attnOut" + suffix,
                                new MatVec(
                                        weight(TensorRole.ATTENTION_OUTPUT),
                                        activation(XB),
                                        activation(XB2),
                                        dim,
                                        dim,
                                        weightDataType)),
                        leaf(
                                "attnResidual" + suffix,
                                new ResidualAdd(
                                        activation(X),
                                        activation(XB2),
                                        activation(X),
                                        DataType.F32)),
                        leaf(
                                "ffnNorm" + suffix,
                                new RmsNorm(
                                        activation(X),
                                        Optional.of(weight(TensorRole.FFN_NORM)),
                                        activation(XB),
                                        config.rmsNormEps(),
                                        DataType.F32)),
                        leaf(
                                "gate" + suffix,
                                new MatVec(
                                        weight(TensorRole.FFN_GATE),
                                        activation(XB),
                                        activation(HB),
                                        config.hiddenDim(),
                                        dim,
                                        weightDataType)),
                        leaf(
                                "up" + suffix,
                                new MatVec(
                                        weight(TensorRole.FFN_UP),
                                        activation(XB),
                                        activation(HB2),
                                        config.hiddenDim(),
                                        dim,
                                        weightDataType)),
                        leaf(
                                "swiglu" + suffix,
                                new SwiGLU(
                                        activation(HB),
                                        activation(HB2),
                                        activation(HB),
                                        DataType.F32)),
                        leaf(
                                "down" + suffix,
                                new MatVec(
                                        weight(TensorRole.FFN_DOWN),
                                        activation(HB),
                                        activation(XB),
                                        dim,
                                        config.hiddenDim(),
                                        weightDataType)),
                        leaf(
                                "ffnResidual" + suffix,
                                new ResidualAdd(
                                        activation(X),
                                        activation(XB),
                                        activation(X),
                                        DataType.F32)));
        return new ProgramComponent.Composite("layer" + suffix, inner, BOTH);
    }

    /**
     * The binding surface.
     *
     * <p>Deliberately coarse for this slice: the weights as one entry per role, the key/value
     * store, the workspace, one control array and one result carrier. It is enough to make the two
     * things that matter true — the key/value dtype reaches the signature, and the invocation
     * values name carriers rather than arrays — without inventing a per-buffer inventory the
     * lowering does not yet consume.
     */
    private static List<BindingEntry> bindings(
            Configuration config,
            DataType weightDataType,
            DataType kvDataType,
            boolean deviceSample) {
        List<BindingEntry> entries = new ArrayList<>();
        int index = 0;
        entries.add(
                new BindingEntry.ProgramFixed(
                        index++,
                        BindingRole.WEIGHT,
                        "weights",
                        weightDataType,
                        Shape.of(config.numberOfLayers(), config.dim()),
                        Direction.IN));
        entries.add(
                new BindingEntry.ProgramFixed(
                        index++,
                        BindingRole.KV_POOL,
                        KEY_STORE,
                        kvDataType,
                        Shape.of(config.numberOfLayers(), config.contextLength(), config.kvDim()),
                        Direction.IN_OUT));
        entries.add(
                new BindingEntry.ProgramFixed(
                        index++,
                        BindingRole.KV_POOL,
                        VALUE_STORE,
                        kvDataType,
                        Shape.of(config.numberOfLayers(), config.contextLength(), config.kvDim()),
                        Direction.IN_OUT));
        entries.add(
                new BindingEntry.ProgramFixed(
                        index++,
                        BindingRole.WORKSPACE,
                        "activations",
                        DataType.F32,
                        Shape.of(config.dim()),
                        Direction.IN_OUT));
        int control = index;
        entries.add(
                new BindingEntry.ProgramFixed(
                        index++,
                        BindingRole.CONTROL,
                        CONTROL,
                        DataType.F32,
                        Shape.of(2),
                        Direction.IN_OUT));
        int logits = index;
        entries.add(
                new BindingEntry.ProgramFixed(
                        index++,
                        BindingRole.RESULT,
                        LOGITS,
                        DataType.F32,
                        Shape.of(config.vocabularySize()),
                        Direction.OUT));
        entries.add(
                new BindingEntry.InvocationValue(
                        index++, ValueId.TOKEN, ValueType.I32, control, 0, 1));
        entries.add(
                new BindingEntry.InvocationValue(
                        index++, ValueId.POSITION, ValueType.I32, control, 1, 1));
        entries.add(
                new BindingEntry.HostVisibleResult(
                        index++, ResultId.LOGITS, null, logits, 0, config.vocabularySize()));
        if (deviceSample) {
            int sampled = index;
            entries.add(
                    new BindingEntry.ProgramFixed(
                            index++,
                            BindingRole.RESULT,
                            SAMPLED,
                            DataType.F32,
                            Shape.of(1),
                            Direction.OUT));
            entries.add(
                    new BindingEntry.HostVisibleResult(
                            index, ResultId.SAMPLED_TOKEN, ValueType.I32, sampled, 0, 1));
        }
        return entries;
    }

    private static ProgramComponent leaf(
            String name, org.beehive.jitllm.program.op.Operation operation) {
        return new ProgramComponent.Leaf(name, operation, BOTH);
    }

    private static OperandRef.Weight weight(TensorRole role) {
        return new OperandRef.Weight(role.name().toLowerCase(java.util.Locale.ROOT), role);
    }

    private static OperandRef activation(String name) {
        return new OperandRef.Activation(name);
    }
}
