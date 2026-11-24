package org.beehive.jitllm.inference.state;

import org.beehive.jitllm.backend.tornado.workspace.KdaBatchArrays;
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

        if (state.prefillBatchWidth > 1) {
            allocateBatch(state, c);
        }

        // [0] = position, [1] = table-local KV slot.
        ws.positionHolder = TornadoWorkspaces.ints(2);
        int local = state.localSize;
        ws.temp = TornadoWorkspaces.floats(1 + ((dim + local - 1) / local));
        ws.tempFFN = TornadoWorkspaces.floats(1 + ((dim + local - 1) / local));
        ws.tempLogits = TornadoWorkspaces.floats(1 + ((dim + local - 1) / local));
    }

    /**
     * The batched prefill's chunk-wide arrays, the chunk padded to whole 128-row tiles: the
     * projections' outputs, the attention's (token, head) rows in FP32 and FP16, the int8 GEMMs'
     * quantized activation, the routed and shared experts' activations, and the KDA rows.
     */
    private static void allocateBatch(KimiLinearState state, KimiLinearConfiguration c) {
        TornadoWorkspace ws = state.workspace;
        int p = (state.prefillBatchWidth + 127) & ~127;
        state.paddedBatch = p;
        state.paddedPositions = (c.contextLength() + 127) & ~127;
        int heads = c.numberOfHeads();
        int keyWidth = c.keyWidth();
        int rank = c.kvLoraRank();
        int dim = c.dim();
        int rows = p * heads;
        int positions = state.paddedPositions;

        ws.wrapNormedBatch = TornadoWorkspaces.floats(p * dim);
        ws.compressedKvBatch = TornadoWorkspaces.floats(p * KimiLinearState.KV_A_PADDED);
        ws.queryBatch = TornadoWorkspaces.floats(p * c.queryDim());
        ws.queryF16Batch = TornadoWorkspaces.halfFloats(p * c.queryDim());
        ws.absorbedBatch = TornadoWorkspaces.floats(rows * keyWidth);
        ws.absorbedF16Batch = TornadoWorkspaces.halfFloats(rows * keyWidth);
        ws.keysF16 = TornadoWorkspaces.halfFloats(positions * keyWidth);
        ws.latentTF16 = TornadoWorkspaces.halfFloats(rank * positions);
        ws.scoresBatch = TornadoWorkspaces.floats(rows * positions);
        ws.probsF16Batch = TornadoWorkspaces.halfFloats(rows * positions);
        ws.latentBatch = TornadoWorkspaces.floats(rows * rank);
        ws.latentF16Batch = TornadoWorkspaces.halfFloats(rows * rank);
        ws.attnOutBatch = TornadoWorkspaces.floats(p * c.attentionOutputInputDim());
        ws.gateBatch = TornadoWorkspaces.floats(p * c.hiddenDim());
        ws.hiddenBatch = TornadoWorkspaces.floats(p * c.hiddenDim());

        int widest =
                Math.max(
                        Math.max(dim, c.hiddenDim()),
                        Math.max(c.attentionOutputInputDim(), c.kdaDim()));
        ws.wrapQ8ActBatch = TornadoWorkspaces.bytes(p * widest);
        ws.wrapQ8ActScales = TornadoWorkspaces.floats(p * widest / 32);
        ws.wrapQ8SplitPartial = TornadoWorkspaces.floats(8 * p * dim);

        int experts = Math.max(1, c.expertCount());
        int assignments = p * Math.max(1, c.expertsUsed());
        int hidden = Math.max(32, c.expertHiddenDim());
        int shared = Math.max(32, c.sharedHiddenDim());
        ws.wrapMoeLogitsBatch = TornadoWorkspaces.floats(p * experts);
        ws.wrapMoeIdsBatch = TornadoWorkspaces.ints(assignments);
        ws.wrapMoeWeightsBatch = TornadoWorkspaces.floats(assignments);
        ws.wrapMoeSharedGateBatch = TornadoWorkspaces.floats(p);
        ws.wrapMoeSortedToken = TornadoWorkspaces.ints(assignments);
        ws.wrapMoePosition = TornadoWorkspaces.ints(assignments);
        ws.wrapMoeTiles =
                TornadoWorkspaces.ints(
                        1
                                + 3
                                        * org.beehive.jitllm.backend.tornado.kernels
                                                .Qwen35MoeBatchKernels.maxTiles(
                                                assignments, experts));
        ws.wrapMoeHiddenBatch = TornadoWorkspaces.floats(assignments * hidden);
        ws.wrapMoeHiddenQ8 = TornadoWorkspaces.bytes(assignments * hidden);
        ws.wrapMoeHiddenScales = TornadoWorkspaces.floats(assignments * hidden / 32);
        ws.wrapMoeOutBatch = TornadoWorkspaces.floats(assignments * dim);
        ws.wrapMoeSharedGateUp = TornadoWorkspaces.floats(p * shared);
        ws.wrapMoeSharedHidden = TornadoWorkspaces.floats(p * shared);
        ws.wrapMoeSharedQ8 = TornadoWorkspaces.bytes(p * shared);
        ws.wrapMoeSharedScales = TornadoWorkspaces.floats(p * shared / 32);
        ws.wrapMoeSharedOut = TornadoWorkspaces.floats(p * dim);

        ws.kdaBatch = KdaBatchArrays.allocate(p, c.kdaHeads(), c.kdaHeadDim());
    }
}
