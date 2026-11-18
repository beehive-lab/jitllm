package org.beehive.jitllm.inference.state;

import org.beehive.jitllm.backend.tornado.workspace.KdaDeviceArrays;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspace;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jitllm.model.kimilinear.KimiLinearConfiguration;

// @formatter:off
/**
 * The device arrays of a {@code kimi-linear} decode plan.
 *
 * <ul>
 *   <li>the latent cache, {@code [MLA layer][position][keyWidth]}: the normalized latent, then the
 *       unrotated key tail; FP16 when the session keeps its cache in half precision;
 *   <li>the KDA windows, {@code [KDA layer][q, k, v][channel][kernel - 1]}, and matrices, {@code
 *       [KDA layer][head][key][value]};
 *   <li>the activations of the projections, the attention and the experts.
 * </ul>
 *
 * <p>Every stage of a split keeps the whole model's layout and uses its own layers' part.
 */
// @formatter:on
final class KimiLinearDeviceWorkspace {

    private KimiLinearDeviceWorkspace() {}

    static void allocate(KimiLinearState state, KimiLinearConfiguration c) {
        TornadoWorkspace ws = state.workspace;
        int dim = c.dim();
        int context = c.contextLength();
        int width = c.kdaDim();
        int hidden = Math.max(c.hiddenDim(), c.sharedHiddenDim());
        int attnOut = c.attentionOutputInputDim();
        int widest = Math.max(Math.max(dim, hidden), Math.max(width, attnOut));

        TornadoWorkspaces.activationQ8_0(ws, dim);
        ws.wrapX = TornadoWorkspaces.floats(dim);
        ws.wrapXb = TornadoWorkspaces.floats(Math.max(dim, Math.max(attnOut, width)));
        ws.wrapXb2 = TornadoWorkspaces.floats(dim);
        TornadoWorkspaces.packedActivation(ws, widest);
        ws.wrapHb = TornadoWorkspaces.floats(hidden);
        ws.wrapLogits = TornadoWorkspaces.floats(c.vocabularySize());

        ws.wrapQ = TornadoWorkspaces.floats(c.queryDim());
        ws.wrapCompressedKv = TornadoWorkspaces.floats(c.compressedKeyValueDim());
        ws.wrapAbsorbedQuery = TornadoWorkspaces.floats(c.absorbedQueryDim());
        ws.wrapLatentOut = TornadoWorkspaces.floats(c.numberOfHeads() * c.kvLoraRank());
        ws.wrapAtt = TornadoWorkspaces.floats(c.numberOfHeads() * context);
        ws.wrapAttSplit =
                TornadoWorkspaces.floats(
                        c.numberOfHeads()
                                * KimiLinearConfiguration.DECODE_ATTENTION_SPLITS
                                * (c.kvLoraRank() + 2));
        int cacheElements = Math.max(1, c.mlaLayerCount() * context * c.keyWidth());
        if (state.storageOptions().usesFp16KeyValueCache()) {
            ws.wrapKeyCacheFP16 = TornadoWorkspaces.halfFloats(cacheElements);
            ws.wrapValueCacheFP16 = TornadoWorkspaces.halfFloats(1);
        } else {
            ws.wrapKeyCache = TornadoWorkspaces.floats(cacheElements);
            ws.wrapValueCache = TornadoWorkspaces.floats(1);
        }
        ws.wrapConvState =
                TornadoWorkspaces.floats(
                        Math.max(1, c.kdaLayerCount() * 3 * width * (c.kdaConvKernel() - 1)));
        ws.wrapDeltaState =
                TornadoWorkspaces.floats(
                        Math.max(
                                1,
                                c.kdaLayerCount()
                                        * c.kdaHeads()
                                        * c.kdaHeadDim()
                                        * c.kdaHeadDim()));
        ws.kda = KdaDeviceArrays.allocate(dim, c.kdaHeads(), c.kdaHeadDim());

        int experts = Math.max(1, c.expertCount());
        int used = Math.max(1, c.expertsUsed());
        ws.wrapRouterLogits = TornadoWorkspaces.floats(experts);
        ws.wrapSelectedExperts = TornadoWorkspaces.ints(used);
        ws.wrapRoutingWeights = TornadoWorkspaces.floats(used);
        int moeHidden = Math.max(32, c.expertsUsed() * c.expertHiddenDim() + c.sharedHiddenDim());
        ws.wrapMoeHidden = TornadoWorkspaces.floats(moeHidden);
        ws.wrapMoeHiddenQuants = TornadoWorkspaces.ints(moeHidden / 4);
        ws.wrapMoeHiddenQScales = TornadoWorkspaces.floats(moeHidden / 32);
        ws.wrapMoeHiddenQSums = TornadoWorkspaces.ints(moeHidden / 32);

        // [0] = position, [1] = table-local KV slot.
        ws.positionHolder = TornadoWorkspaces.ints(2);
        int local = state.localSize;
        ws.temp = TornadoWorkspaces.floats(1 + ((dim + local - 1) / local));
        ws.tempFFN = TornadoWorkspaces.floats(1 + ((dim + local - 1) / local));
        ws.tempLogits = TornadoWorkspaces.floats(1 + ((dim + local - 1) / local));
    }
}
