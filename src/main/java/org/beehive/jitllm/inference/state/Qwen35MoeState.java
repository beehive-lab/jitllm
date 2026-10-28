package org.beehive.jitllm.inference.state;

import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspaces;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.qwen35moe.Qwen35Experts;
import org.beehive.jitllm.tensor.standard.ArrayFloatTensor;
import org.beehive.jitllm.tensor.standard.FloatTensor;

// @formatter:off
/**
 * The state of a {@code qwen35moe} session: the {@code qwen35} state, plus the routing and expert
 * activations of the mixture-of-experts feed-forward.
 *
 * <p>The host arrays are for the CPU forward pass. The device ones are allocated where the {@code
 * qwen35} state allocated its own device workspace, and the chunk-wide ones where it sized that
 * workspace for a batched prefill.
 */
// @formatter:on
public final class Qwen35MoeState extends Qwen35State {

    /** Router scores, one per expert; softmaxed in place by the router. */
    public final FloatTensor expertScores;

    /** The experts selected for the current token, highest weight first. */
    public final int[] expertIds;

    /** Their renormalized routing weights, in the same order. */
    public final float[] expertWeights;

    /** One expert's (or the shared expert's) gate and up activations. */
    public final FloatTensor expertHidden;

    public final FloatTensor expertHiddenUp;

    /** One expert's output, before it is weighted into the feed-forward result. */
    public final FloatTensor expertOut;

    public Qwen35MoeState(Configuration config, Qwen35Experts experts, int batchsize) {
        super(config, batchsize);
        int width = Math.max(experts.hiddenDim(), experts.sharedHiddenDim());
        this.expertScores = ArrayFloatTensor.allocate(experts.count());
        this.expertIds = new int[experts.used()];
        this.expertWeights = new float[experts.used()];
        this.expertHidden = ArrayFloatTensor.allocate(width);
        this.expertHiddenUp = ArrayFloatTensor.allocate(width);
        this.expertOut = ArrayFloatTensor.allocate(config.dim());
        if (workspace.wrapX != null) {
            allocateDeviceExperts(experts);
            if (prefillBatchWidth > 1) {
                allocateBatchExperts(workspace, config, experts, prefillBatchWidth);
            }
        }
    }

    /**
     * The routing of one token and its experts' activations; the shared expert's gate is one value.
     */
    private void allocateDeviceExperts(Qwen35Experts experts) {
        workspace.wrapRouterLogits = TornadoWorkspaces.floats(experts.count());
        workspace.wrapSelectedExperts = TornadoWorkspaces.ints(experts.used());
        workspace.wrapRoutingWeights = TornadoWorkspaces.floats(experts.used());
        workspace.wrapSharedGate = TornadoWorkspaces.floats(1);
        int moeHidden = experts.routedHiddenDim() + experts.sharedHiddenDim();
        workspace.wrapMoeHidden = TornadoWorkspaces.floats(moeHidden);
        workspace.wrapMoeHiddenQuants = TornadoWorkspaces.ints(moeHidden / 4);
        workspace.wrapMoeHiddenQScales = TornadoWorkspaces.floats(moeHidden / 32);
        workspace.wrapMoeHiddenQSums = TornadoWorkspaces.ints(moeHidden / 32);
    }

    /**
     * A chunk's routing, its assignments sorted by expert, and the experts' activations in that
     * sorted order; the shared expert's over the whole chunk. Shared with the other families whose
     * batched feed-forward runs {@code Qwen35MoeBatchFeedForward}'s expert tasks.
     */
    static void allocateBatchExperts(
            org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspace workspace,
            Configuration config,
            Qwen35Experts experts,
            int batch) {
        int assignments = batch * experts.used();
        int hidden = experts.hiddenDim();
        int shared = Math.max(experts.sharedHiddenDim(), 32);
        workspace.wrapMoeLogitsBatch = TornadoWorkspaces.floats(batch * experts.count());
        workspace.wrapMoeIdsBatch = TornadoWorkspaces.ints(assignments);
        workspace.wrapMoeWeightsBatch = TornadoWorkspaces.floats(assignments);
        workspace.wrapMoeSharedGateBatch = TornadoWorkspaces.floats(batch);
        workspace.wrapMoeSortedToken = TornadoWorkspaces.ints(assignments);
        workspace.wrapMoePosition = TornadoWorkspaces.ints(assignments);
        workspace.wrapMoeTiles =
                TornadoWorkspaces.ints(
                        1
                                + 3
                                        * org.beehive.jitllm.backend.tornado.kernels
                                                .Qwen35MoeBatchKernels.maxTiles(
                                                assignments, experts.count()));
        workspace.wrapMoeHiddenBatch = TornadoWorkspaces.floats(assignments * hidden);
        workspace.wrapMoeHiddenQ8 = TornadoWorkspaces.bytes(assignments * hidden);
        workspace.wrapMoeHiddenScales = TornadoWorkspaces.floats(assignments * hidden / 32);
        workspace.wrapMoeOutBatch = TornadoWorkspaces.floats(assignments * config.dim());
        workspace.wrapMoeSharedGateUp = TornadoWorkspaces.floats(batch * shared);
        workspace.wrapMoeSharedHidden = TornadoWorkspaces.floats(batch * shared);
        workspace.wrapMoeSharedQ8 = TornadoWorkspaces.bytes(batch * shared);
        workspace.wrapMoeSharedScales = TornadoWorkspaces.floats(batch * shared / 32);
        workspace.wrapMoeSharedOut = TornadoWorkspaces.floats(batch * config.dim());
        TornadoWorkspaces.sharedExpertTiles(
                workspace,
                batch,
                org.beehive.jitllm.backend.tornado.kernels.Qwen35MoeBatchKernels.TILE_ROWS);
    }
}
