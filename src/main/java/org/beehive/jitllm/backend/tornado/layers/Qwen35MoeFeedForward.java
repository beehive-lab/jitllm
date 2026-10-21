package org.beehive.jitllm.backend.tornado.layers;

import java.util.List;
import java.util.function.UnaryOperator;
import org.beehive.jitllm.backend.tornado.kernels.Qwen35MoeKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.backend.tornado.workspace.TornadoWorkspace;
import org.beehive.jitllm.inference.weights.Qwen35ExpertWeights;
import org.beehive.jitllm.model.qwen35moe.Qwen35Experts;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;

// @formatter:off
/**
 * The {@code qwen35moe} feed-forward of one token, in place of the dense one in {@link
 * Qwen35FFNLayers}: from the normalized and quantized {@code wrapXb} into the residual stream.
 *
 * <p>Router and shared gate, top-k, every selected expert's and the shared expert's gate/up, and
 * their weighted down projections added to {@code wrapX}. See {@link Qwen35MoeKernels}. The layer
 * graphs own the graph structure; this class adds the tasks, names the weights and scratch they
 * bind, and registers the tasks' grids.
 */
// @formatter:on
public final class Qwen35MoeFeedForward implements Qwen35FeedForward {

    private final TornadoWorkspace workspace;
    private final Qwen35Experts experts;
    private final Qwen35ExpertWeights<TornadoTensor> tensors;
    private final int dim;

    /**
     * @param experts the experts' shape
     * @param tensors their tensors, per layer
     * @param dim the model width
     */
    public Qwen35MoeFeedForward(
            TornadoWorkspace workspace,
            Qwen35Experts experts,
            Qwen35ExpertWeights<TornadoTensor> tensors,
            int dim) {
        this.workspace = workspace;
        this.experts = experts;
        this.tensors = tensors;
        this.dim = dim;
    }

    /** Adds layer {@code layerIndex}'s feed-forward tasks, each named through {@code tn}. */
    @Override
    public void appendTasks(
            TaskGraph layer, KernelContext context, int layerIndex, UnaryOperator<String> tn) {
        layer.task(
                tn.apply("moe_router"),
                Qwen35MoeKernels::routerAndSharedGate,
                context,
                workspace.wrapXb,
                tensors.router()[layerIndex].asFloatArray(),
                tensors.sharedGateInput()[layerIndex].asFloatArray(),
                workspace.wrapRouterLogits,
                workspace.wrapSharedGate,
                dim,
                experts.count());
        layer.task(
                tn.apply("moe_topk"),
                Qwen35MoeKernels::routerTopK,
                context,
                workspace.wrapRouterLogits,
                workspace.wrapSelectedExperts,
                workspace.wrapRoutingWeights,
                experts.count(),
                experts.used());
        layer.task(
                tn.apply("moe_gate_up"),
                Qwen35MoeKernels::expertsGateUpQ8_0DP4A,
                context,
                workspace.wrapXbQuants,
                workspace.wrapXbScales,
                workspace.wrapSelectedExperts,
                tensors.gateExperts()[layerIndex].asByteArray(),
                tensors.upExperts()[layerIndex].asByteArray(),
                tensors.sharedGate()[layerIndex].asByteArray(),
                tensors.sharedUp()[layerIndex].asByteArray(),
                workspace.wrapMoeHidden,
                dim,
                experts.hiddenDim(),
                experts.sharedHiddenDim(),
                experts.used());
        layer.task(
                tn.apply("moe_down_quantize"),
                TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                context,
                workspace.wrapMoeHidden,
                workspace.wrapMoeHiddenQuants,
                workspace.wrapMoeHiddenQScales,
                workspace.wrapMoeHiddenQSums);
        layer.task(
                tn.apply("moe_down"),
                Qwen35MoeKernels::expertsDownResidualQ8_0DP4A,
                context,
                workspace.wrapMoeHiddenQuants,
                workspace.wrapMoeHiddenQScales,
                workspace.wrapSelectedExperts,
                workspace.wrapRoutingWeights,
                workspace.wrapSharedGate,
                tensors.downExperts()[layerIndex].asByteArray(),
                tensors.sharedDown()[layerIndex].asByteArray(),
                workspace.wrapX,
                dim,
                experts.hiddenDim(),
                experts.sharedHiddenDim(),
                experts.used());
    }

    /** Layer {@code layerIndex}'s expert tensors, in place of the dense gate, up and down. */
    @Override
    public List<Object> layerWeights(int layerIndex) {
        return List.of(
                tensors.router()[layerIndex],
                tensors.sharedGateInput()[layerIndex],
                tensors.gateExperts()[layerIndex],
                tensors.upExperts()[layerIndex],
                tensors.downExperts()[layerIndex],
                tensors.sharedGate()[layerIndex],
                tensors.sharedUp()[layerIndex],
                tensors.sharedDown()[layerIndex]);
    }

    /** The routing and expert scratch, bound by every layer graph in turn. */
    @Override
    public Object[] scratch() {
        return new Object[] {
            workspace.wrapRouterLogits,
            workspace.wrapSelectedExperts,
            workspace.wrapRoutingWeights,
            workspace.wrapSharedGate,
            workspace.wrapMoeHidden,
            workspace.wrapMoeHiddenQuants,
            workspace.wrapMoeHiddenQScales,
            workspace.wrapMoeHiddenQSums
        };
    }

    /** Grids of one layer's feed-forward tasks, whose qualified names start with {@code prefix}. */
    @Override
    public void addWorkerGrids(GridScheduler scheduler, String prefix) {
        int hidden = experts.routedHiddenDim() + experts.sharedHiddenDim();
        scheduler.addWorkerGrid(prefix + "moe_router", warpWorker(experts.count() + 1));
        scheduler.addWorkerGrid(prefix + "moe_topk", WorkerGridFactory.genericWorker(32, 32));
        scheduler.addWorkerGrid(prefix + "moe_gate_up", warpWorker(hidden));
        scheduler.addWorkerGrid(
                prefix + "moe_down_quantize", WorkerGridFactory.genericWorker(hidden, 32));
        scheduler.addWorkerGrid(prefix + "moe_down", warpWorker(dim));
    }

    /** Lanes for {@code warps} warps, rounded up to whole {@link Qwen35MoeKernels#LOCAL} blocks. */
    private static WorkerGrid warpWorker(int warps) {
        int lanes = warps * 32;
        int local = Qwen35MoeKernels.LOCAL;
        return WorkerGridFactory.genericWorker((lanes + local - 1) / local * local, local);
    }
}
