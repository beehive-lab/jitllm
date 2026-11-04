package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.backend.tornado.ops.LayerProjections;
import org.beehive.jitllm.backend.tornado.ops.Operands;
import org.beehive.jitllm.backend.tornado.ops.Region;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.jitllm.model.qwen35.Qwen35Configuration;
import org.beehive.jitllm.program.op.MatMul;
import org.beehive.jitllm.program.op.OperandRef;
import org.beehive.jitllm.program.op.Operation;
import org.beehive.jitllm.program.op.ResidualAdd;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.TensorRole;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Qwen3.5's batched-prefill projections, one region each, so that the plan picks for each whether
 * cuBLAS or the JIT kernels run it ({@link LayerProjections}). The attention, the recurrent scan
 * and the norms stay the layer class's own tasks.
 *
 * <p>Example: layer 0 of Qwen3.5-0.8B is recurrent, so its regions are ssm_qkv, ssm_gate,
 * ssm_alpha, ssm_beta and ssm_out, then gate, up and down.
 *
 * <p>Activation names: {@link #NORMED} the normed chunk, {@link #IN} any other input, {@link #OUT}
 * the output, and {@link #STREAM} the residual stream that a residual projection adds into.
 */
public final class Qwen35PrefillRegions {

    public static final String NORMED = "normed";
    public static final String IN = "in";
    public static final String OUT = "out";
    public static final String STREAM = "stream";

    private Qwen35PrefillRegions() {}

    /**
     * One projection: its task, its weight's name in messages, its role, its weights by layer, its
     * outputs and inputs, whether it reads the normed chunk, and whether it adds into the residual
     * stream.
     */
    public record Projection(
            String task,
            String name,
            TensorRole role,
            TornadoTensor[] tensors,
            int outputs,
            int inputs,
            boolean normedInput,
            boolean residual) {

        /** Layer {@code layer}'s weight, or null where the layer has none. */
        public TornadoTensor tensor(int layer) {
            return tensors != null && layer < tensors.length ? tensors[layer] : null;
        }
    }

    /** Every projection a Qwen3.5 layer can have. An attention layer has q, k, v and o. */
    public static List<Projection> projections(
            Qwen35Configuration config, Qwen35TornadoWeights weights) {
        int dim = config.dim();
        int kvDim = config.kvDim();
        int valueDim = config.deltaNetValueDim();
        TensorRole other = TensorRole.OTHER;
        return List.of(
                new Projection(
                        "attn_q_proj",
                        "attn_q",
                        TensorRole.ATTENTION_QUERY,
                        weights.wqLayered,
                        config.queryGateDim(),
                        dim,
                        true,
                        false),
                new Projection(
                        "attn_k_proj",
                        "attn_k",
                        TensorRole.ATTENTION_KEY,
                        weights.wkLayered,
                        kvDim,
                        dim,
                        true,
                        false),
                new Projection(
                        "attn_v_proj",
                        "attn_v",
                        TensorRole.ATTENTION_VALUE,
                        weights.wvLayered,
                        kvDim,
                        dim,
                        true,
                        false),
                new Projection(
                        "attn_output_proj",
                        "attn_output",
                        TensorRole.ATTENTION_OUTPUT,
                        weights.woLayered,
                        dim,
                        config.attentionOutputInputDim(),
                        false,
                        true),
                new Projection(
                        "ffn_gate_proj",
                        "ffn_gate",
                        TensorRole.FFN_GATE,
                        weights.w1Layered,
                        config.hiddenDim(),
                        dim,
                        true,
                        false),
                new Projection(
                        "ffn_up_proj",
                        "ffn_up",
                        TensorRole.FFN_UP,
                        weights.w3Layered,
                        config.hiddenDim(),
                        dim,
                        true,
                        false),
                new Projection(
                        "ffn_down_proj",
                        "ffn_down",
                        TensorRole.FFN_DOWN,
                        weights.w2Layered,
                        dim,
                        config.hiddenDim(),
                        false,
                        true),
                new Projection(
                        "ssm_qkv_proj",
                        "attn_qkv",
                        other,
                        weights.ssmQkv,
                        config.deltaNetConvDim(),
                        dim,
                        true,
                        false),
                new Projection(
                        "ssm_gate_proj",
                        "attn_gate",
                        other,
                        weights.ssmGate,
                        valueDim,
                        dim,
                        true,
                        false),
                new Projection(
                        "ssm_alpha_proj",
                        "ssm_alpha",
                        other,
                        weights.ssmAlpha,
                        config.numberOfValueHeads(),
                        dim,
                        true,
                        false),
                new Projection(
                        "ssm_beta_proj",
                        "ssm_beta",
                        other,
                        weights.ssmBeta,
                        config.numberOfValueHeads(),
                        dim,
                        true,
                        false),
                new Projection(
                        "ssm_out_proj",
                        "ssm_out",
                        other,
                        weights.ssmOut,
                        dim,
                        valueDim,
                        false,
                        true));
    }

    /**
     * {@code p}'s region in {@code layer}, which must have its weight: the projection, then the
     * residual add for a residual projection.
     */
    public static Region region(Projection p, int layer, int tokens) {
        List<Operation> ops = new ArrayList<>(2);
        ops.add(
                new MatMul(
                        new OperandRef.Weight(p.name(), p.role()),
                        a(p.normedInput() ? NORMED : IN),
                        a(OUT),
                        p.outputs(),
                        p.inputs(),
                        tokens,
                        p.tensor(layer).dataType()));
        if (p.residual()) {
            ops.add(new ResidualAdd(a(STREAM), a(OUT), a(STREAM), DataType.F32));
        }
        return new Region(layer, p.task(), tokens, ops);
    }

    /**
     * The arrays of one projection call: {@code x} its input, {@code out} its output (the stream,
     * for a residual projection), {@code w} its weight.
     */
    public static Operands operands(FloatArray x, FloatArray out, TornadoTensor w) {
        return new Operands() {
            @Override
            public Object activation(String name) {
                return switch (name) {
                    case NORMED, IN -> x;
                    case OUT, STREAM -> out;
                    default ->
                            throw new IllegalArgumentException("no array for activation " + name);
                };
            }

            @Override
            public TornadoTensor weight(OperandRef.Weight ref, int layer) {
                return w;
            }
        };
    }

    private static OperandRef.Activation a(String name) {
        return new OperandRef.Activation(name);
    }
}
