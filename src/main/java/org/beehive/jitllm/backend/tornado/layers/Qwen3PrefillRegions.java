package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.function.BiFunction;
import org.beehive.jitllm.backend.tornado.ops.Region;
import org.beehive.jitllm.model.architecture.Qwen3ProgramDescription;
import org.beehive.jitllm.model.qwen3.Qwen3Configuration;
import org.beehive.jitllm.program.op.MatMul;
import org.beehive.jitllm.program.op.OperandRef;
import org.beehive.jitllm.program.op.Operation;
import org.beehive.jitllm.program.op.ResidualAdd;
import org.beehive.jitllm.program.op.RmsNorm;
import org.beehive.jitllm.program.op.SwiGLU;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.TensorRole;

/**
 * The operations of a Qwen3 layer that the batched prefill runs through implementations: the ones
 * before attention and the ones after it. They are the operations of {@code
 * Qwen3ProgramDescription}'s layer, with each projection over a chunk of tokens.
 *
 * <p>Activation names: {@code x} the residual stream, {@code attnIn} and {@code ffnIn} the normed
 * inputs, {@code q}, {@code k}, {@code v}, {@code attn} the attention output, {@code o}, {@code
 * gate}, {@code up}, {@code hidden} the SwiGLU output, and {@code down}. A layer class maps them to
 * its buffers.
 */
public final class Qwen3PrefillRegions {

    private Qwen3PrefillRegions() {}

    /** One layer's two regions. */
    public record Layer(Region beforeAttention, Region afterAttention) {}

    private static OperandRef.Activation a(String name) {
        return new OperandRef.Activation(name);
    }

    private static OperandRef.Weight w(TensorRole role) {
        return new OperandRef.Weight(role.name().toLowerCase(Locale.ROOT), role);
    }

    /**
     * Every layer's regions.
     *
     * @param tokens the tokens of one chunk
     * @param rows the rows the projections are sized for: {@code tokens}, or more where the GEMM
     *     needs padding
     * @param type the type of each weight (Q8_0, Q4_K, ...) by role and layer
     */
    public static List<Layer> of(
            Qwen3Configuration config,
            int tokens,
            int rows,
            BiFunction<TensorRole, Integer, DataType> type) {
        int dim = config.dim();
        int qDim = config.queryDim();
        int kvDim = config.kvDim();
        int hidden = config.hiddenDim();
        float eps = config.rmsNormEps();
        List<Layer> layers = new ArrayList<>(config.numberOfLayers());
        for (int i = 0; i < config.numberOfLayers(); i++) {
            int layer = i;
            List<Operation> before =
                    List.of(
                            new RmsNorm(
                                    a("x"),
                                    Optional.of(w(TensorRole.ATTENTION_NORM)),
                                    a("attnIn"),
                                    eps,
                                    DataType.F32,
                                    1,
                                    dim),
                            projection(
                                    TensorRole.ATTENTION_QUERY,
                                    "attnIn",
                                    "q",
                                    qDim,
                                    dim,
                                    rows,
                                    type,
                                    layer),
                            projection(
                                    TensorRole.ATTENTION_KEY,
                                    "attnIn",
                                    "k",
                                    kvDim,
                                    dim,
                                    rows,
                                    type,
                                    layer),
                            projection(
                                    TensorRole.ATTENTION_VALUE,
                                    "attnIn",
                                    "v",
                                    kvDim,
                                    dim,
                                    rows,
                                    type,
                                    layer));
            List<Operation> after =
                    List.of(
                            projection(
                                    TensorRole.ATTENTION_OUTPUT,
                                    "attn",
                                    "o",
                                    dim,
                                    qDim,
                                    rows,
                                    type,
                                    layer),
                            new ResidualAdd(a("x"), a("o"), a("x"), DataType.F32),
                            new RmsNorm(
                                    a("x"),
                                    Optional.of(w(TensorRole.FFN_NORM)),
                                    a("ffnIn"),
                                    eps,
                                    DataType.F32,
                                    1,
                                    dim),
                            projection(
                                    TensorRole.FFN_GATE,
                                    "ffnIn",
                                    "gate",
                                    hidden,
                                    dim,
                                    rows,
                                    type,
                                    layer),
                            projection(
                                    TensorRole.FFN_UP,
                                    "ffnIn",
                                    "up",
                                    hidden,
                                    dim,
                                    rows,
                                    type,
                                    layer),
                            new SwiGLU(a("gate"), a("up"), a("hidden"), DataType.F32),
                            projection(
                                    TensorRole.FFN_DOWN,
                                    "hidden",
                                    "down",
                                    dim,
                                    hidden,
                                    rows,
                                    type,
                                    layer),
                            new ResidualAdd(a("x"), a("down"), a("x"), DataType.F32));
            layers.add(
                    new Layer(
                            new Region(i, "before attention", tokens, before),
                            new Region(i, "after attention", tokens, after)));
        }
        return layers;
    }

    /**
     * Every layer's attention, one region each, between the two {@link Layer} regions: from q, k
     * and v into {@code attn}.
     */
    public static List<Region> attention(Qwen3Configuration config, int tokens) {
        List<Region> regions = new ArrayList<>(config.numberOfLayers());
        for (int i = 0; i < config.numberOfLayers(); i++) {
            regions.add(
                    new Region(
                            i,
                            "attention",
                            tokens,
                            List.of(
                                    Qwen3ProgramDescription.attention(
                                            config,
                                            a("q"),
                                            a("k"),
                                            a("v"),
                                            a("attn"),
                                            DataType.F32))));
        }
        return regions;
    }

    /** All regions of {@code layers}, in the order the layer graphs run them. */
    public static List<Region> all(List<Layer> layers) {
        List<Region> regions = new ArrayList<>(2 * layers.size());
        for (Layer layer : layers) {
            regions.add(layer.beforeAttention());
            regions.add(layer.afterAttention());
        }
        return regions;
    }

    /** {@code input} times the layer's weight with {@code role}: {@code outputs} values per row. */
    private static MatMul projection(
            TensorRole role,
            String input,
            String output,
            int outputs,
            int inputs,
            int rows,
            BiFunction<TensorRole, Integer, DataType> type,
            int layer) {
        return new MatMul(
                w(role), a(input), a(output), outputs, inputs, rows, type.apply(role, layer));
    }
}
