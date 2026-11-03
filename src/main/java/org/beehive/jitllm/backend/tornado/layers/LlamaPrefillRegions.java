package org.beehive.jitllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.function.BiFunction;
import org.beehive.jitllm.backend.tornado.ops.Region;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.program.op.MatMul;
import org.beehive.jitllm.program.op.OperandRef;
import org.beehive.jitllm.program.op.Operation;
import org.beehive.jitllm.program.op.ResidualAdd;
import org.beehive.jitllm.program.op.RmsNorm;
import org.beehive.jitllm.program.op.SwiGLU;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.runtime.tensor.TensorRole;

/**
 * The operations of a Llama layer that the tensor-core batched prefill runs through
 * implementations, split where a family runs tasks of its own between them. Llama's layers are also
 * run by Mistral, Devstral, Qwen2, DeepSeek-R1-Distill-Qwen, Phi-3 and Granite.
 *
 * <p>Example: Granite scales the down projection's output before it is added to the stream, so the
 * down projection and that residual add are two regions, and Granite's scale task runs between
 * them.
 *
 * <p>Activation names, as in {@link Qwen3PrefillRegions}: {@code x} the residual stream, {@code
 * attnIn} and {@code ffnIn} the normed inputs, {@code q}, {@code k}, {@code v} (or {@code qkv}),
 * {@code attn} the attention output, {@code o}, {@code gate}, {@code up} (or {@code gateUp}),
 * {@code hidden} the SwiGLU output, and {@code down}. A layer class maps them to its buffers.
 */
public final class LlamaPrefillRegions {

    private LlamaPrefillRegions() {}

    /**
     * One layer's regions, in the order they run: the attention RMS norm and q|k|v; the attention
     * output projection; the residual add, the feed-forward RMS norm and gate|up; the down
     * projection; its residual add. SwiGLU, between gate|up and down, stays the layer's own task.
     */
    public record Layer(
            Region beforeAttention,
            Region attentionOutput,
            Region feedForwardInput,
            Region down,
            Region residual) {

        List<Region> all() {
            return List.of(beforeAttention, attentionOutput, feedForwardInput, down, residual);
        }
    }

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
     * @param rows the rows the projections are sized for: {@code tokens} rounded up to whole GEMM
     *     tiles
     * @param fused whether q|k|v and gate|up are one weight each, as in Phi-3
     * @param denseFeedForward whether the layer has the dense gate, up and down projections; a
     *     mixture of experts (Qwen2-MoE) runs its own feed-forward instead, and its three
     *     feed-forward regions are empty
     * @param type the type of each weight (Q8_0, Q4_K, ...) by role and layer
     */
    public static List<Layer> of(
            Configuration config,
            int tokens,
            int rows,
            boolean fused,
            boolean denseFeedForward,
            BiFunction<TensorRole, Integer, DataType> type) {
        int dim = config.dim();
        int qDim = config.numberOfHeads() * config.headSize();
        int kvDim = config.kvDim();
        int hidden = config.hiddenDim();
        float eps = config.rmsNormEps();
        List<Layer> layers = new ArrayList<>(config.numberOfLayers());
        for (int i = 0; i < config.numberOfLayers(); i++) {
            int layer = i;
            List<Operation> beforeAttention = new ArrayList<>();
            beforeAttention.add(
                    new RmsNorm(
                            a("x"),
                            Optional.of(w(TensorRole.ATTENTION_NORM)),
                            a("attnIn"),
                            eps,
                            DataType.F32,
                            1,
                            dim));
            if (fused) {
                beforeAttention.add(
                        projection(
                                TensorRole.ATTENTION_QKV,
                                "attnIn",
                                "qkv",
                                qDim + 2 * kvDim,
                                dim,
                                rows,
                                type,
                                layer));
            } else {
                beforeAttention.add(
                        projection(
                                TensorRole.ATTENTION_QUERY,
                                "attnIn",
                                "q",
                                qDim,
                                dim,
                                rows,
                                type,
                                layer));
                beforeAttention.add(
                        projection(
                                TensorRole.ATTENTION_KEY,
                                "attnIn",
                                "k",
                                kvDim,
                                dim,
                                rows,
                                type,
                                layer));
                beforeAttention.add(
                        projection(
                                TensorRole.ATTENTION_VALUE,
                                "attnIn",
                                "v",
                                kvDim,
                                dim,
                                rows,
                                type,
                                layer));
            }
            List<Operation> attentionOutput =
                    List.of(
                            projection(
                                    TensorRole.ATTENTION_OUTPUT,
                                    "attn",
                                    "o",
                                    dim,
                                    qDim,
                                    rows,
                                    type,
                                    layer));
            List<Operation> feedForwardInput = new ArrayList<>();
            List<Operation> down = new ArrayList<>();
            List<Operation> residual = new ArrayList<>();
            if (denseFeedForward) {
                feedForwardInput.add(new ResidualAdd(a("x"), a("o"), a("x"), DataType.F32));
                feedForwardInput.add(
                        new RmsNorm(
                                a("x"),
                                Optional.of(w(TensorRole.FFN_NORM)),
                                a("ffnIn"),
                                eps,
                                DataType.F32,
                                1,
                                dim));
                if (fused) {
                    feedForwardInput.add(
                            projection(
                                    TensorRole.FFN_GATE_UP,
                                    "ffnIn",
                                    "gateUp",
                                    2 * hidden,
                                    dim,
                                    rows,
                                    type,
                                    layer));
                } else {
                    feedForwardInput.add(
                            projection(
                                    TensorRole.FFN_GATE,
                                    "ffnIn",
                                    "gate",
                                    hidden,
                                    dim,
                                    rows,
                                    type,
                                    layer));
                    feedForwardInput.add(
                            projection(
                                    TensorRole.FFN_UP,
                                    "ffnIn",
                                    "up",
                                    hidden,
                                    dim,
                                    rows,
                                    type,
                                    layer));
                }
                down.add(
                        projection(
                                TensorRole.FFN_DOWN,
                                "hidden",
                                "down",
                                dim,
                                hidden,
                                rows,
                                type,
                                layer));
                residual.add(new ResidualAdd(a("x"), a("down"), a("x"), DataType.F32));
            }
            layers.add(
                    new Layer(
                            new Region(i, "before attention", tokens, beforeAttention),
                            new Region(i, "attention output", tokens, attentionOutput),
                            new Region(i, "feed-forward input", tokens, feedForwardInput),
                            new Region(i, "down", tokens, down),
                            new Region(i, "residual", tokens, residual)));
        }
        return layers;
    }

    /**
     * One layer's operations before and after attention, with SwiGLU among them: the shape MLX and
     * the Metal JIT kernels merge into a few steps (RMS norm + q|k|v; the output projection with
     * its residual add; the feed-forward block).
     */
    public record Whole(Region beforeAttention, Region afterAttention) {}

    /** Every layer's operations as {@link Whole} layers; the arguments as {@link #of}. */
    public static List<Whole> whole(
            Configuration config,
            int tokens,
            int rows,
            boolean fused,
            BiFunction<TensorRole, Integer, DataType> type) {
        List<Whole> layers = new ArrayList<>(config.numberOfLayers());
        for (Layer layer : of(config, tokens, rows, fused, true, type)) {
            List<Operation> after = new ArrayList<>(layer.attentionOutput().ops());
            after.addAll(layer.feedForwardInput().ops());
            after.add(
                    fused
                            ? new SwiGLU(a("gateUp"), a("gateUp"), a("hidden"), DataType.F32)
                            : new SwiGLU(a("gate"), a("up"), a("hidden"), DataType.F32));
            after.addAll(layer.down().ops());
            after.addAll(layer.residual().ops());
            Region before = layer.beforeAttention();
            layers.add(
                    new Whole(
                            before, new Region(before.layer(), "after attention", tokens, after)));
        }
        return layers;
    }

    /** All regions of {@code layers}, in the order the layer graphs run them. */
    public static List<Region> allWhole(List<Whole> layers) {
        List<Region> regions = new ArrayList<>(2 * layers.size());
        for (Whole layer : layers) {
            regions.add(layer.beforeAttention());
            regions.add(layer.afterAttention());
        }
        return regions;
    }

    /** All regions of {@code layers}, in the order the layer graphs run them. */
    public static List<Region> all(List<Layer> layers) {
        List<Region> regions = new ArrayList<>(5 * layers.size());
        for (Layer layer : layers) {
            regions.addAll(layer.all());
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
