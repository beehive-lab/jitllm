package org.beehive.jitllm.model.loader;

import java.nio.channels.FileChannel;
import java.util.Map;
import org.beehive.jitllm.auxiliary.Pair;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.format.GGMLTensorEntry;
import org.beehive.jitllm.format.GGUF;
import org.beehive.jitllm.inference.weights.Qwen35ExpertWeights;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.inference.weights.standard.Qwen35MoeStandardWeights;
import org.beehive.jitllm.inference.weights.standard.Qwen35StandardWeights;
import org.beehive.jitllm.inference.weights.tornado.Qwen35MoeTornadoWeights;
import org.beehive.jitllm.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.jitllm.model.qwen35.Qwen35;
import org.beehive.jitllm.model.qwen35.Qwen35Configuration;
import org.beehive.jitllm.model.qwen35moe.Qwen35Experts;
import org.beehive.jitllm.model.qwen35moe.Qwen35Moe;
import org.beehive.jitllm.tensor.standard.FloatTensor;
import org.beehive.jitllm.tokenizer.Tokenizer;

// @formatter:off
/**
 * Loads a {@code qwen35moe} model: the {@code qwen35} trunk, read by {@link Qwen35ModelLoader}
 * under this architecture's keys, without its dense feed-forward, and the experts in its place.
 *
 * <p>The trunk's hidden width is the routed experts' combined width for one token, which is what
 * every buffer sized by it holds.
 */
// @formatter:on
public class Qwen35MoeModelLoader extends Qwen35ModelLoader {

    /** Read with the configuration, before the weights and the model need it. */
    private Qwen35Experts experts;

    public Qwen35MoeModelLoader(
            FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Qwen35Configuration createConfiguration(Map<String, Object> metadata) {
        String p = metadata.get("general.architecture") + ".";
        this.experts =
                new Qwen35Experts(
                        (int) metadata.get(p + "expert_count"),
                        (int) metadata.get(p + "expert_used_count"),
                        (int) metadata.get(p + "expert_feed_forward_length"),
                        metadata.containsKey(p + "expert_shared_feed_forward_length")
                                ? (int) metadata.get(p + "expert_shared_feed_forward_length")
                                : 0);
        return super.createConfiguration(metadata);
    }

    @Override
    protected int feedForwardLength(Map<String, Object> metadata, String prefix) {
        return experts.routedHiddenDim();
    }

    @Override
    protected boolean hasDenseFeedForward() {
        return false;
    }

    @Override
    protected Qwen35 createModel(Qwen35Configuration config, Tokenizer tokenizer, Weights weights) {
        Qwen35 trunk = super.createModel(config, tokenizer, weights);
        return new Qwen35Moe(config, experts, tokenizer, weights, trunk.chatFormat());
    }

    @Override
    protected Weights createStandardWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            Qwen35Configuration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        var trunk =
                (Qwen35StandardWeights)
                        super.createStandardWeights(
                                tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        int blocks = config.numberOfBlocks();
        return new Qwen35MoeStandardWeights(
                trunk,
                new Qwen35ExpertWeights<FloatTensor>(
                        perBlock(blocks, l -> layer(tensorEntries, l, "ffn_gate_inp")),
                        perBlock(blocks, l -> layer(tensorEntries, l, "ffn_gate_exps")),
                        perBlock(blocks, l -> layer(tensorEntries, l, "ffn_up_exps")),
                        perBlock(blocks, l -> layer(tensorEntries, l, "ffn_down_exps")),
                        perBlock(blocks, l -> layer(tensorEntries, l, "ffn_gate_shexp")),
                        perBlock(blocks, l -> layer(tensorEntries, l, "ffn_up_shexp")),
                        perBlock(blocks, l -> layer(tensorEntries, l, "ffn_down_shexp")),
                        perBlock(blocks, l -> layer(tensorEntries, l, "ffn_gate_inp_shexp"))));
    }

    @Override
    protected Weights createTornadoVMWeights(
            Map<String, GGMLTensorEntry> tensorEntries,
            Qwen35Configuration config,
            Pair<float[], float[]> ropeFreqs,
            GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        var trunk =
                (Qwen35TornadoWeights)
                        super.createTornadoVMWeights(
                                tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        int blocks = config.numberOfBlocks();
        return new Qwen35MoeTornadoWeights(
                trunk,
                new Qwen35ExpertWeights<TornadoTensor>(
                        perBlockDevice(blocks, l -> layer(tensorEntries, l, "ffn_gate_inp")),
                        perBlockDevice(blocks, l -> layer(tensorEntries, l, "ffn_gate_exps")),
                        perBlockDevice(blocks, l -> layer(tensorEntries, l, "ffn_up_exps")),
                        perBlockDevice(blocks, l -> layer(tensorEntries, l, "ffn_down_exps")),
                        perBlockDevice(blocks, l -> layer(tensorEntries, l, "ffn_gate_shexp")),
                        perBlockDevice(blocks, l -> layer(tensorEntries, l, "ffn_up_shexp")),
                        perBlockDevice(blocks, l -> layer(tensorEntries, l, "ffn_down_shexp")),
                        perBlockDevice(
                                blocks, l -> layer(tensorEntries, l, "ffn_gate_inp_shexp"))));
    }

    private static GGMLTensorEntry layer(
            Map<String, GGMLTensorEntry> entries, int layer, String kind) {
        return entries.get("blk." + layer + "." + kind + ".weight");
    }
}
