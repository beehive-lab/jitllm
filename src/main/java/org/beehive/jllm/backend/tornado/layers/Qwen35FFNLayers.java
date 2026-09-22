package org.beehive.jllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jllm.backend.tornado.device.TornadoDevices;
import org.beehive.jllm.backend.tornado.kernels.Qwen35AttentionKernels;
import org.beehive.jllm.backend.tornado.kernels.Qwen35DeltaNetKernels;
import org.beehive.jllm.backend.tornado.kernels.Qwen3Kernels;
import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsQ4_1;
import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsQ4_K;
import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsQ5_K;
import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernelsQ6_K;
import org.beehive.jllm.backend.tornado.kernels.TransformerPagedKvKernels;
import org.beehive.jllm.backend.tornado.plan.FusedOperandSupport;
import org.beehive.jllm.backend.tornado.scheduling.SchedulerType;
import org.beehive.jllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jllm.inference.state.Qwen35State;
import org.beehive.jllm.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.jllm.model.qwen35.Qwen35Configuration;
import org.beehive.jllm.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * The {@code qwen35} transformer layers, one task graph per block.
 *
 * <h2>Two layer kinds, one graph shape</h2>
 *
 * <p>Three quarters of the blocks mix with a Gated Delta Net and one quarter with attention, and
 * which is which is {@link Qwen35Configuration#isRecurrentLayer(int)} — the same answer the host
 * path uses. Both kinds then run the same dense SwiGLU feed-forward, and both have the same outer
 * residual shape. The per-layer arrays are indexed by <b>absolute</b> block, and are {@code null}
 * at blocks of the other kind; a layer validates the tensors its own kind needs before it builds a
 * single task, so a mis-shaped file fails naming the layer, the role and the representation rather
 * than by dereferencing null somewhere inside TornadoVM.
 *
 * <h2>Every task is selected by the representation of the tensor it reads</h2>
 *
 * <p>This is the first family whose model is genuinely mixed: Q4_0 projections and embeddings, Q4_1
 * {@code ffn_down} on the first eight blocks, Q5_K {@code ssm_out}, a Q6_K vocabulary projection,
 * F32 norms and SSM parameters. Nothing is materialized to a common representation, so a task is
 * bound to a format-specific kernel chosen <b>here</b>, before compilation — never to a kernel that
 * switches on a dtype inside its inner loop, which would cost the compiler the fixed addressing
 * that makes the loop worth writing.
 *
 * <p>The one task that reads two weights at once is the fused gate/up feed-forward. It states that
 * its operands must share a representation and refuses a mixture by name; it does not convert one
 * of them, and it does not read one block layout as another.
 *
 * <h2>What it does not do</h2>
 *
 * <p>Single-token decode only. There are no prefill or batch variants of these graphs, and the
 * provider declares none. The MTP block past the trunk is not built here: it is a draft head, not
 * part of ordinary generation, and {@code numberOfLayers()} excludes it.
 */
// @formatter:on
public class Qwen35FFNLayers
        extends AbstractTransformerLayerTaskGraphs<Qwen35TornadoWeights, Qwen35Configuration> {

    /** Lanes per workgroup for a matrix-vector task: one workgroup reduces one output row. */
    private static final int MATVEC_LOCAL = 128;

    /** Lanes per workgroup for an elementwise task. */
    private static final int ELEMENTWISE_LOCAL = 128;

    /** The head width the split-KV warp kernel is written for: eight dimensions per lane. */
    private static final int SPLIT_KV_MAX_HEAD = 256;

    /** Lanes per split-KV workgroup: one warp per (head, split). */
    private static final int SPLIT_KV_LOCAL = 32;

    private final Qwen35State qwen35State;

    /**
     * What each weight-reading task was bound to, in construction order.
     *
     * <p>Recorded rather than derived: which kernel a task got is what changes when a tensor's
     * representation changes, and it is the only thing about a built plan a test can compare
     * without executing it. {@code Qwen35GraphTopologyTest} asserts over this.
     */
    private final List<Dispatch> dispatches = new ArrayList<>();

    // @formatter:off
    /**
     * One weight-reading task: where it is, what it reads, which kernel decodes it, and whether its
     * <b>activation</b> reached it quantized.
     *
     * <p>The last of those is not derivable from the other three. Whether a projection takes the
     * packed-integer path depends on what the buffer it reads happens to hold at that point in the
     * layer, which is a fact about ordering rather than about the tensor — so it is recorded here
     * and asserted, not inferred.
     */
    // @formatter:on
    public record Dispatch(
            int layer,
            String task,
            String role,
            DataType representation,
            boolean quantizedActivation) {}

    /**
     * The graph layer 0 consumes its activation from.
     *
     * <p>{@code activationUpdate} in the single-token plan and {@code decodeActivation} in the
     * prefill/decode one. The layer computation is identical in both — sequential prefill is these
     * graphs with the logits graph skipped — so the plan shape is the only thing that differs, and
     * it differs by a name.
     */
    private final String activationGraphName;

    public Qwen35FFNLayers(
            String taskGraphName,
            Qwen35State state,
            Qwen35TornadoWeights weights,
            Qwen35Configuration config,
            SchedulerType schedulerType) {
        this(taskGraphName, state, weights, config, schedulerType, "activationUpdate");
    }

    public Qwen35FFNLayers(
            String taskGraphName,
            Qwen35State state,
            Qwen35TornadoWeights weights,
            Qwen35Configuration config,
            SchedulerType schedulerType,
            String activationGraphName) {
        super(taskGraphName, state, weights, config, schedulerType);
        this.qwen35State = state;
        this.activationGraphName = activationGraphName;
        setupFFNLayers();
    }

    /** The dispatch inventory, in construction order. */
    public List<Dispatch> dispatchInventory() {
        return List.copyOf(dispatches);
    }

    // ── validation ────────────────────────────────────────────────────────────

    /**
     * The tensor a layer's role must carry, or a failure naming what is missing.
     *
     * <p>A {@code null} here means the file disagrees with the topology the metadata declares — a
     * block the configuration calls recurrent that carries attention tensors, or the reverse. That
     * is a load-time fact, and the message says which block and which role rather than leaving a
     * null to surface as a NullPointerException inside graph construction.
     */
    private TornadoTensor require(TornadoTensor[] tensors, int layer, String role) {
        TornadoTensor tensor = tensors == null || layer >= tensors.length ? null : tensors[layer];
        if (tensor == null) {
            throw new IllegalStateException(
                    "qwen35 block "
                            + layer
                            + " is a "
                            + (config.isRecurrentLayer(layer) ? "recurrent" : "attention")
                            + " layer and must carry "
                            + role
                            + ", which this model does not hold for it");
        }
        return tensor;
    }

    // ── per-representation matrix-vector dispatch ─────────────────────────────

    /**
     * {@code out = w · x}, or {@code out += w · x}, by the representation {@code w} is in.
     *
     * <p>The selection happens here, at plan construction, so each task is compiled against one
     * block layout with fixed addressing. A representation with no kernel for this shape is refused
     * by name: converting it would double what it occupies and hide the gap.
     */
    private void matVec(
            TaskGraph graph,
            int layer,
            String task,
            String role,
            TornadoTensor w,
            FloatArray x,
            FloatArray out,
            int n,
            int d,
            boolean residual) {
        requireWholeBlocks(layer, task, role, w.dataType(), n);
        boolean packed =
                w.dataType() == DataType.Q4_0
                        && ((!residual && x == state.workspace.wrapXb && normedActivationQuantized)
                                || (residual
                                        && x == state.workspace.wrapHb
                                        && hiddenActivationQuantized));
        // ffn_down where this family holds it as Q4_1 -- the first eight blocks. The activation is
        // the one ffn_down_quantize already produced for the Q4_0 layers, so the condition is the
        // Q4_0 residual case's with the representation changed; the weights stay native Q4_1.
        boolean packedQ4_1 =
                w.dataType() == DataType.Q4_1
                        && residual
                        && x == state.workspace.wrapHb
                        && hiddenActivationQuantized;
        // The Q5_K readout projection, computed once and used both to record the dispatch and to
        // choose the kernel below: a flag that decided one and not the other would make the
        // inventory describe a plan that was not built. Eligibility is carried entirely by
        // ssmActivationQuantized, which is set only where the quantization was actually emitted --
        // that is, only under the packed gate, only in a recurrent layer, and only when the scratch
        // was long enough for the readout.
        boolean packedQ5_K =
                w.dataType() == DataType.Q5_K
                        && residual
                        && x == state.workspace.wrapSsmOut
                        && ssmActivationQuantized;
        dispatches.add(
                new Dispatch(layer, task, role, w.dataType(), packed || packedQ5_K || packedQ4_1));
        switch (w.dataType()) {
            case F32 -> {
                if (residual) {
                    throw unsupported(layer, task, role, w.dataType(), "an accumulating");
                }
                graph.task(
                        tn(task),
                        TransformerComputeKernelsLayered::matrixVectorGeneric,
                        context,
                        x,
                        out,
                        w.asFloatArray(),
                        n,
                        d,
                        MATVEC_LOCAL);
            }
            case F16 -> {
                if (residual) {
                    graph.task(
                            tn(task),
                            TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                            context,
                            x,
                            out,
                            w.asHalfFloatArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            tn(task),
                            TransformerComputeKernelsLayered::matrixVectorGeneric,
                            context,
                            x,
                            out,
                            w.asHalfFloatArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                }
            }
            case Q8_0 -> {
                if (residual) {
                    graph.task(
                            tn(task),
                            TransformerComputeKernelsLayered
                                    ::matrixVectorGenericWithResidualQ8_0Byte,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            tn(task),
                            TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                }
            }
            case Q4_0 -> {
                if (residual && x == state.workspace.wrapHb && hiddenActivationQuantized) {
                    graph.task(
                            tn(task),
                            TransformerComputeKernelsQ4_0::matrixVectorGenericWithResidualQ4_0DP4A,
                            context,
                            state.workspace.wrapXbQuants,
                            state.workspace.wrapXbScales,
                            state.workspace.wrapXbSums,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else if (!residual && x == state.workspace.wrapXb && normedActivationQuantized) {
                    // The packed-integer path, for a projection whose input the branch already
                    // quantized. Weights stay Q4_0; the activation is what changed representation.
                    graph.task(
                            tn(task),
                            TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0DP4A,
                            context,
                            state.workspace.wrapXbQuants,
                            state.workspace.wrapXbScales,
                            state.workspace.wrapXbSums,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else if (residual) {
                    graph.task(
                            tn(task),
                            TransformerComputeKernelsQ4_0::matrixVectorGenericWithResidualQ4_0,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            tn(task),
                            TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                }
            }
            case Q4_1 -> {
                if (packedQ4_1) {
                    // The same contract the Q4_0 branch above uses: the activation this projection
                    // reads was quantized into the shared scratch by ffn_down_quantize, which
                    // already runs for every layer. Weights stay native Q4_1; only the activation
                    // changed representation.
                    graph.task(
                            tn(task),
                            TransformerComputeKernelsQ4_1::matrixVectorGenericWithResidualQ4_1DP4A,
                            context,
                            state.workspace.wrapXbQuants,
                            state.workspace.wrapXbScales,
                            state.workspace.wrapXbSums,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else if (residual) {
                    graph.task(
                            tn(task),
                            TransformerComputeKernelsQ4_1::matrixVectorGenericWithResidualQ4_1,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            tn(task),
                            TransformerComputeKernelsQ4_1::matrixVectorGenericQ4_1,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                }
            }
            case Q4_K -> {
                if (residual) {
                    graph.task(
                            tn(task),
                            TransformerComputeKernelsQ4_K::matrixVectorGenericWithResidualQ4_K,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            tn(task),
                            TransformerComputeKernelsQ4_K::matrixVectorGenericQ4_K,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                }
            }
            case Q5_K -> {
                if (packedQ5_K) {
                    // The packed-integer path for ssm_out, whose activation the delta-net branch
                    // has just quantized. Weights stay Q5_K; the activation is what changed
                    // representation.
                    graph.task(
                            tn(task),
                            TransformerComputeKernelsQ5_K::matrixVectorGenericWithResidualQ5_KDP4A,
                            context,
                            state.workspace.wrapXbQuants,
                            state.workspace.wrapXbScales,
                            state.workspace.wrapXbSums,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else if (residual) {
                    graph.task(
                            tn(task),
                            TransformerComputeKernelsQ5_K::matrixVectorGenericWithResidualQ5_K,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            tn(task),
                            TransformerComputeKernelsQ5_K::matrixVectorGenericQ5_K,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                }
            }
            case Q6_K -> {
                if (residual) {
                    graph.task(
                            tn(task),
                            TransformerComputeKernelsQ6_K::matrixVectorGenericWithResidualQ6_K,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            tn(task),
                            TransformerComputeKernelsQ6_K::matrixVectorGenericQ6_K,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                }
            }
            default -> throw unsupported(layer, task, role, w.dataType(), "a");
        }
    }

    // @formatter:off
    /**
     * A quantized row must be a whole number of blocks.
     *
     * <p>Every block-decoding kernel here addresses a row as {@code row * blocksPerRow} blocks, so
     * it assumes each row starts on a block boundary. That holds for every projection in a real
     * file — a quantizer will not split a block across rows — and when it does not hold the kernel
     * reads a neighbouring row's blocks, producing weights of plausible magnitude and fluent, wrong
     * output. Checked here because the alternative is discovering it as a numerical disagreement on
     * a model that takes minutes to load.
     */
    // @formatter:on
    private void requireWholeBlocks(int layer, String task, String role, DataType type, int n) {
        int blockSize =
                switch (type) {
                    case Q4_0, Q4_1, Q8_0 -> 32;
                    case Q4_K, Q5_K, Q6_K -> 256;
                    default -> 1;
                };
        if (n % blockSize != 0) {
            throw new UnsupportedOperationException(
                    "qwen35 layer "
                            + layer
                            + " task '"
                            + task
                            + "' reads "
                            + role
                            + " as "
                            + type
                            + " with a row of "
                            + n
                            + " weights, which is not a whole number of "
                            + blockSize
                            + "-weight blocks. Every block-decoding kernel addresses a row by its"
                            + " block offset, so a partial row would read the next row's blocks.");
        }
    }

    private UnsupportedOperationException unsupported(
            int layer, String task, String role, DataType type, String article) {
        return new UnsupportedOperationException(
                "qwen35 layer "
                        + layer
                        + " task '"
                        + task
                        + "' reads "
                        + role
                        + " as "
                        + type
                        + ", for which this backend has no "
                        + article
                        + " matrix-vector kernel. It is not converted to another representation to"
                        + " get one: that would hide a missing kernel behind a memory cost and a"
                        + " silent change of arithmetic.");
    }

    /**
     * The fused gate/up feed-forward, by the representation both weights share.
     *
     * <p>The one task here that decodes two weight matrices in one pass, so it is the one task
     * whose operands must agree. They are checked rather than assumed: a Q4_0 kernel handed a Q5_K
     * operand reads 176-byte super-blocks as ten 18-byte blocks and produces weights of plausible
     * magnitude.
     */
    private void fusedGateUp(
            TaskGraph graph, int layer, TornadoTensor gate, TornadoTensor up, FloatArray x) {
        FusedOperandSupport.requireUniform(
                "qwen35 layer " + layer + " fused gate/up feed-forward",
                List.of("ffn_gate", "ffn_up"),
                gate,
                up);
        boolean packed =
                gate.dataType() == DataType.Q4_0
                        && x == qwen35State.workspace.wrapXb
                        && normedActivationQuantized;
        dispatches.add(
                new Dispatch(layer, "ffn_gate_up", "ffn_gate|ffn_up", gate.dataType(), packed));
        if (packed) {
            graph.task(
                    tn("ffn_gate_up"),
                    TransformerComputeKernelsQ4_0::fusedFFNGateUpSiLUQ4_0DP4A,
                    context,
                    qwen35State.workspace.wrapXbQuants,
                    qwen35State.workspace.wrapXbScales,
                    qwen35State.workspace.wrapXbSums,
                    qwen35State.workspace.wrapHb,
                    gate.asByteArray(),
                    up.asByteArray(),
                    config.dim(),
                    config.hiddenDim(),
                    MATVEC_LOCAL);
            return;
        }
        switch (gate.dataType()) {
            case Q4_0 ->
                    graph.task(
                            tn("ffn_gate_up"),
                            TransformerComputeKernelsQ4_0::fusedFFNGateUpSiLUQ4_0,
                            context,
                            x,
                            qwen35State.workspace.wrapHb,
                            gate.asByteArray(),
                            up.asByteArray(),
                            config.dim(),
                            config.hiddenDim(),
                            MATVEC_LOCAL);
            case Q4_K ->
                    graph.task(
                            tn("ffn_gate_up"),
                            TransformerComputeKernelsQ4_K::fusedFFNGateUpSiLUQ4_K,
                            context,
                            x,
                            qwen35State.workspace.wrapHb,
                            gate.asByteArray(),
                            up.asByteArray(),
                            config.dim(),
                            config.hiddenDim(),
                            MATVEC_LOCAL);
            case Q8_0 ->
                    graph.task(
                            tn("ffn_gate_up"),
                            TransformerComputeKernelsLayered
                                    ::fusedFeedForwardWithSiLUAndGLUActivationQ8_0Byte,
                            context,
                            x,
                            qwen35State.workspace.wrapHb,
                            gate.asByteArray(),
                            up.asByteArray(),
                            config.dim(),
                            config.hiddenDim(),
                            MATVEC_LOCAL);
            case F16 ->
                    graph.task(
                            tn("ffn_gate_up"),
                            TransformerComputeKernelsLayered
                                    ::fusedFeedForwardWithSiLUAndGLUActivation,
                            context,
                            x,
                            qwen35State.workspace.wrapHb,
                            gate.asHalfFloatArray(),
                            up.asHalfFloatArray(),
                            config.dim(),
                            config.hiddenDim(),
                            MATVEC_LOCAL);
            default ->
                    throw unsupported(
                            layer, "ffn_gate_up", "ffn_gate|ffn_up", gate.dataType(), "a");
        }
    }

    // ── the layer graphs ──────────────────────────────────────────────────────

    // @formatter:off
    /**
     * The prefix the tasks of {@code layerIndex} carry inside their graph.
     *
     * <p>Empty while a layer owns its graph, which is every family's default: the graph name
     * already separates one layer's {@code attn_rms_reduce} from the next one's. A family that puts
     * more than one layer in a graph must return something distinct per layer, because grid keys
     * are {@code graphName.taskName} and two layers would otherwise collide on every task.
     */
    // @formatter:on
    protected String layerTaskPrefix(int layerIndex) {
        return "";
    }

    /**
     * The graph {@code layerIndex}'s tasks belong to. One per layer unless a family groups them.
     */
    protected String layerGraphName(int layerIndex) {
        return "layer_" + layerIndex;
    }

    /**
     * Whether {@code layerIndex} is the first layer of its graph, and so owns the graph's inputs.
     */
    protected final boolean firstLayerOfGraph(int layerIndex) {
        return layerIndex == 0
                || !layerGraphName(layerIndex - 1).equals(layerGraphName(layerIndex));
    }

    /** Whether {@code layerIndex} is the last layer of its graph, and so publishes its outputs. */
    protected final boolean lastLayerOfGraph(int layerIndex) {
        return layerIndex == config.numberOfLayers() - 1
                || !layerGraphName(layerIndex + 1).equals(layerGraphName(layerIndex));
    }

    /** The prefix in force while the current layer's tasks are being added. */
    private String taskPrefix = "";

    /** Qualifies a task name with the layer it belongs to, when its graph holds more than one. */
    private String tn(String name) {
        return taskPrefix + name;
    }

    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        return appendLayer(new TaskGraph(layerGraphName(layerIndex)), layerIndex);
    }

    // @formatter:off
    /**
     * Adds one transformer layer's tasks to {@code layer}, in order.
     *
     * <p>Separate from graph creation so a family can put more than one layer in a graph. The
     * inputs a graph consumes and the outputs it persists belong to the graph, not to each layer,
     * so they are guarded by {@link #firstLayerOfGraph} and {@link #lastLayerOfGraph}; everything
     * between is per layer and unchanged.
     */
    // @formatter:on
    protected TaskGraph appendLayer(TaskGraph layer, int layerIndex) {
        taskPrefix = layerTaskPrefix(layerIndex);

        // Only when the producer is a different graph. Within one graph the previous layer's
        // store of wrapX is an ordinary dependency between tasks and consuming it would be wrong.
        if (firstLayerOfGraph(layerIndex)) {
            String predecessor =
                    layerIndex == 0 ? activationGraphName : layerGraphName(layerIndex - 1);
            layer.consumeFromDevice(predecessor, qwen35State.workspace.wrapX);
            configureLayerDataTransfers(layer, layerIndex);
        }
        transferLayerWeights(layer, layerIndex);

        // Input normalization, shared by both mixers: xb = attn_norm ⊙ rms(x).
        // The apply carries the quantization every Q4_0 projection below reads, in one task,
        // where the capability holds. Without it the apply is the plain elementwise one and
        // nothing is quantized.
        normalize(
                layer,
                "attn_rms_reduce",
                "attn_rms_finalize",
                "attn_rms_apply",
                qwen35State.workspace.temp,
                require(weights.rms_att_weightLayered, layerIndex, "attn_norm"),
                DP4A);
        normedActivationQuantized = DP4A;

        if (config.isRecurrentLayer(layerIndex)) {
            deltaNetBranch(layer, layerIndex);
        } else {
            attentionBranch(layer, layerIndex);
        }

        // The feed-forward's input norm is the file's post_attention_norm; there is no ffn_norm.
        // It writes over wrapXb, so whatever was quantized from it no longer describes it.
        normedActivationQuantized = false;
        ssmActivationQuantized = false;
        // Its own quantization, of the feed-forward's own activation, carried by its own apply.
        // The branch's quants describe the attention norm's output, which this is not. The
        // scratch is the same three arrays: the branch's projections are all behind us in this
        // graph, so the buffers are free, and a second set would cost memory to say the same
        // thing.
        normalize(
                layer,
                "ffn_rms_reduce",
                "ffn_rms_finalize",
                "ffn_rms_apply",
                qwen35State.workspace.tempFFN,
                require(weights.rms_ffn_weightLayered, layerIndex, "post_attention_norm"),
                DP4A);
        if (DP4A) {
            normedActivationQuantized = true;
        }

        fusedGateUp(
                layer,
                layerIndex,
                require(weights.w1Layered, layerIndex, "ffn_gate"),
                require(weights.w3Layered, layerIndex, "ffn_up"),
                qwen35State.workspace.wrapXb);
        hiddenActivationQuantized = false;
        if (DP4A) {
            // SwiGLU's output, quantized fresh. It is neither of the activations quantized
            // earlier in this layer, and the scratch it shares with them is sized for it.
            layer.task(
                    tn("ffn_down_quantize"),
                    TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                    context,
                    qwen35State.workspace.wrapHb,
                    qwen35State.workspace.wrapXbQuants,
                    qwen35State.workspace.wrapXbScales,
                    qwen35State.workspace.wrapXbSums);
            hiddenActivationQuantized = true;
        }

        matVec(
                layer,
                layerIndex,
                "ffn_down_proj",
                "ffn_down",
                require(weights.w2Layered, layerIndex, "ffn_down"),
                qwen35State.workspace.wrapHb,
                qwen35State.workspace.wrapX,
                config.hiddenDim(),
                config.dim(),
                true);

        if (lastLayerOfGraph(layerIndex)) {
            layer.persistOnDevice(
                    qwen35State.workspace.wrapX,
                    keyStore(),
                    valueStore(),
                    qwen35State.workspace.wrapConvState,
                    qwen35State.workspace.wrapDeltaState);
        }
        return layer;
    }

    // @formatter:off
    /**
     * How many KV splits this family's decode attention runs, or one for the per-head kernel.
     *
     * <p>Split-KV decomposes each head's key/value range across {@code DECODE_ATTENTION_SPLITS}
     * warps and a combine pass; the warp kernel walks its split's positions in order, each lane
     * holding eight dimensions of the query, the key, the value and the output in registers, the
     * score a shuffle-down sum. The launch is one warp per (head, split): {@code SPLIT_KV_LOCAL}
     * lanes, and the kernel is written for a 256-wide head ({@code SPLIT_KV_MAX_HEAD}).
     *
     * <p>Restricted to <b>CUDA</b>, whose warp shuffles are the verified ones and where this was
     * measured. FP16 key/value storage is an option rather than a backend property, so the
     * capability alone would have admitted OpenCL, where nothing has run. The FP32 key/value path
     * keeps the kernel it has, on every backend. One is not a special case: it selects the per-head
     * kernel.
     */
    // @formatter:on
    private int attentionSplits() {
        var device = org.beehive.jllm.backend.tornado.device.TornadoDevices.current();
        boolean eligible =
                fp16Kv()
                        && config.headSize() == SPLIT_KV_MAX_HEAD
                        && org.beehive.jllm.runtime.backend.BackendId.CUDA.equals(device.backend())
                        && device.capabilities()
                                .supports(
                                        org.beehive.jllm.runtime.backend.DeviceCapability
                                                .SPLIT_KV_ATTENTION);
        return eligible ? Qwen35Configuration.DECODE_ATTENTION_SPLITS : 1;
    }

    /** Whether this state's key/value store is half precision. */
    protected boolean fp16Kv() {
        return state.usesFp16KeyValueCache();
    }

    /** The key store the graphs bind, whichever precision it is in. */
    protected Object keyStore() {
        return fp16Kv() ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
    }

    protected Object valueStore() {
        return fp16Kv() ? state.workspace.wrapValueCacheFP16 : state.workspace.wrapValueCache;
    }

    // @formatter:off
    /**
     * Whether the delta rule splits each column's reduction across eight lanes.
     *
     * <p>A property of the head's width: the split cuts the rows into eight, so the width must be a
     * multiple of eight, and the workgroup it asks for is eight times that width, which a device
     * must accept (1024 lanes at most). This family's value head is 128 wide, so the workgroup is
     * 1024.
     *
     * <p>Not a user choice and not a tuning knob. A geometry that does not satisfy it keeps the
     * one-lane-per-column kernel.
     */
    // @formatter:on
    private boolean deltaRuleIsSplit() {
        return deltaRuleGeometry() == DeltaRuleGeometry.SPLIT8;
    }

    /** The decode delta-rule kernel and the workgroup it is built for, one decision. */
    public enum DeltaRuleGeometry {
        /** {@code deltaRuleSplit8}: a workgroup of {@code 8 * headDim} lanes per value head. */
        SPLIT8(Qwen35DeltaNetKernels.DELTA_RULE_PARTS),
        /** {@code deltaRuleSplit}: a workgroup of {@code 2 * headDim} lanes per value head. */
        SPLIT2(2),
        /** {@code deltaRule}: a lane per column, the elementwise workgroup. */
        LANE_PER_COLUMN(0);

        final int parts;

        DeltaRuleGeometry(int parts) {
            this.parts = parts;
        }

        /** The workgroup the kernel needs, or 0 for the elementwise default. */
        int localSize(int headDim) {
            return parts * headDim;
        }
    }

    // @formatter:off
    /**
     * Which delta-rule kernel a value head of {@code headDim} columns runs on a device whose
     * largest workgroup is {@code maxWorkGroup} lanes (0 when the runtime did not say).
     *
     * <p>The split kernels map lanes to (part, column) exactly: lane {@code tid} of a workgroup of
     * {@code parts * headDim} is part {@code tid / headDim}, column {@code tid % headDim}, and the
     * parts meet through shared memory inside that workgroup. A runtime that cannot give the
     * workgroup asked for shrinks it, and a shrunk workgroup does not run a slower version of the
     * kernel, it runs a wrong one. So a split geometry is chosen only when the device reports a
     * limit that admits it, in full; an unknown limit admits none. The eight-part kernel is
     * preferred (measured +1.7-1.9% decode over the two-part one on this family's 128-wide head,
     * closer to FP64); the two-part kernel is the fallback it replaced; the lane-per-column kernel
     * needs no assumption beyond the elementwise workgroup and always runs.
     */
    // @formatter:on
    public static DeltaRuleGeometry selectDeltaRuleGeometry(int headDim, long maxWorkGroup) {
        for (DeltaRuleGeometry g :
                new DeltaRuleGeometry[] {DeltaRuleGeometry.SPLIT8, DeltaRuleGeometry.SPLIT2}) {
            if (headDim % g.parts == 0
                    && maxWorkGroup > 0
                    && g.localSize(headDim) <= maxWorkGroup) {
                return g;
            }
        }
        return DeltaRuleGeometry.LANE_PER_COLUMN;
    }

    private DeltaRuleGeometry deltaRuleGeometry() {
        return selectDeltaRuleGeometry(
                config.headValueDim(), TornadoDevices.current().maxWorkGroupSize());
    }

    // @formatter:off
    /**
     * Whether the delta-net L2 norm takes a workgroup per head rather than a lane per head.
     *
     * <p>The same condition {@link #gatedNormIsWide} applies, read on the <b>key</b> head's width
     * because that is what the L2 norm reduces over: the shared tree halves a power-of-two width at
     * every step and its local work size is that width. This model's key head is 128 wide.
     */
    // @formatter:on
    private boolean l2NormIsWide() {
        int headDim = config.headKeyDim();
        return headDim > 0 && (headDim & (headDim - 1)) == 0;
    }

    // @formatter:off
    /**
     * Whether the gated norm takes a workgroup per head rather than a lane per head.
     *
     * <p>A property of the head's width, not a user choice and not a backend one: the wide kernel
     * reduces through a shared tree, which halves a power-of-two width at every step, and its local
     * work size is that width. Anything else keeps the per-head lane. This model's value head is
     * 128 wide.
     */
    // @formatter:on
    private boolean gatedNormIsWide() {
        int headDim = config.headValueDim();
        return headDim > 0 && (headDim & (headDim - 1)) == 0;
    }

    /** {@code xb = weight ⊙ rms(x)} — the reduction, its finalize where needed, and the apply. */
    private void normalize(
            TaskGraph layer,
            String reduce,
            String finalize,
            String apply,
            FloatArray scratch,
            TornadoTensor weight) {
        normalize(layer, reduce, finalize, apply, scratch, weight, false);
    }

    // @formatter:off
    /**
     * The reduce, the optional finalize, and the apply — with the apply folded into the block
     * quantization when {@code quantize} is set.
     *
     * <p>Only the <b>apply</b> is folded. It is elementwise, so the quantization's 32-lane
     * workgroup owns exactly the elements its own lanes would have normalized and nothing
     * synchronizes across a workgroup; the reduce, which does span the row, stays its own task
     * because folding it would need exactly that. {@code wrapXb} is still written — the F32 {@code
     * ssm_alpha} and {@code ssm_beta} projections read it, and so does every non-packed projection
     * — so what goes away is the second launch and the read-back, not the store.
     *
     * <p>The fused task takes the apply's name and the quantization's grid. A caller that folds
     * must not also emit a separate quantization task for the same activation.
     */
    // @formatter:on
    private void normalize(
            TaskGraph layer,
            String reduce,
            String finalize,
            String apply,
            FloatArray scratch,
            TornadoTensor weight,
            boolean quantize) {
        layer.task(
                tn(reduce),
                rmsReduceKernel(),
                context,
                scratch,
                qwen35State.workspace.wrapX,
                config.dim(),
                config.rmsNormEps(),
                qwen35State.localSize);
        if (shouldUseFinalNormalization()) {
            layer.task(
                    tn(finalize),
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    scratch,
                    config.dim(),
                    config.rmsNormEps());
        }
        if (quantize) {
            layer.task(
                    tn(apply),
                    TransformerComputeKernelsQ4_0::rmsApplyAndQuantizeActivationQ8Blocks,
                    context,
                    qwen35State.workspace.wrapXb,
                    qwen35State.workspace.wrapX,
                    weight.asFloatArray(),
                    scratch,
                    qwen35State.workspace.wrapXbQuants,
                    qwen35State.workspace.wrapXbScales,
                    qwen35State.workspace.wrapXbSums);
            return;
        }
        layer.task(
                tn(apply),
                TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                context,
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapX,
                weight.asFloatArray(),
                scratch);
    }

    // @formatter:off
    /**
     * Whether {@code wrapXb} still holds the activation {@code xb_quantize} quantized.
     *
     * <p>The buffer is reused inside a layer — the attention branch writes its gated output into
     * it, and the feed-forward norm writes over it again — so the identity of the array a
     * projection reads says nothing about <b>which</b> activation is in it. This says. It is set
     * where the quantization is emitted and cleared at every point the contents change, and it is
     * what the packed-integer dispatch consults; without it, a projection reading {@code wrapXb}
     * after either of those writes would silently consume the previous activation's quants. Today
     * the one projection that would — {@code attn_output_proj} — is excluded for the unrelated
     * reason that it folds a residual, which is not a property worth depending on.
     */
    // @formatter:on
    private boolean normedActivationQuantized;

    /**
     * Whether {@code wrapHb} holds the activation the feed-forward's own quantization describes.
     *
     * <p>Separate from {@link #normedActivationQuantized} because it is a different buffer holding
     * a different activation: SwiGLU's output, which only {@code ffn_down} reads.
     */
    private boolean hiddenActivationQuantized;

    // @formatter:off
    /**
     * Whether this device's Q4_0 projections read a quantized activation and a packed integer dot
     * product rather than a floating-point one.
     *
     * <p>A device fact, not a user choice: the packed path needs {@code dp4a} to be lowered, and it
     * is worth taking only where that has been measured. Everything else about the decision is a
     * property of the projection — Q4_0 weights, no residual, and an input the branch has already
     * quantized — and is decided where the task is built.
     *
     * <p>Prefill is untouched. What the path costs is in the parity record: the activation is
     * quantized to eight bits, so the logits move by far more than the floating-point path's bounds
     * allow — relative L2 2.45e-2 against a 1e-4 bound, cosine 0.99970 — while the decisions did
     * not, at 0/63 argmax disagreements and token-identical greedy output over 120 tokens.
     */
    // @formatter:on
    private static final boolean DP4A =
            TornadoDevices.current()
                            .capabilities()
                            .supports(
                                    org.beehive.jllm.runtime.backend.DeviceCapability
                                            .PACKED_INTEGER_DOT)
                    // The escape hatch is for the tests whose subject is addressing rather than
                    // arithmetic: they compare the device against the host exactly, which a
                    // quantized activation cannot do. Not a user option, and not a CLI flag.
                    && !"false"
                            .equalsIgnoreCase(
                                    System.getProperty("jllm.qwen35.packedIntegerDot", "true"));

    /**
     * Whether {@code wrapSsmOut} holds the activation {@code ssm_out_quantize} quantized.
     *
     * <p>Separate from the two flags above for the same reason they are separate from each other: a
     * different buffer holding a different activation, with exactly one reader.
     */
    private boolean ssmActivationQuantized;

    /**
     * Whether the shared Q8-block scratch is long enough for an activation of {@code elements}.
     *
     * <p>It is sized for the widest activation the Q4_0 projections read, and the delta-net readout
     * is not one of those, so its width is a fact to check rather than assume. On the 27B it is
     * 6144 against a scratch sized for 17408; a configuration where it were not would otherwise
     * quantize past the end of three arrays.
     */
    private boolean packedScratchHolds(int elements) {
        return elements <= Math.max(config.dim(), config.hiddenDim()) && elements % 32 == 0;
    }

    // @formatter:off
    /**
     * A full-attention block, from the normalized {@code wrapXb} back into {@code wrapX}.
     *
     * <p>Follows the host branch operation for operation. Three things separate it from Qwen3's:
     * the query projection is twice as wide and carries an interleaved output gate; the rotation
     * covers 64 of a 256-wide head; and the attention result is gated by a logistic before the
     * output projection.
     */
    // @formatter:on
    private void attentionBranch(TaskGraph layer, int layerIndex) {
        final int headDim = config.numberOfHeadsKey();
        final int kvDim = config.kvDim();
        final int attnDim = config.attentionOutputInputDim();

        matVec(
                layer,
                layerIndex,
                "attn_q_proj",
                "attn_q",
                require(weights.wqLayered, layerIndex, "attn_q"),
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapQ,
                config.dim(),
                config.queryGateDim(),
                false);
        matVec(
                layer,
                layerIndex,
                "attn_k_proj",
                "attn_k",
                require(weights.wkLayered, layerIndex, "attn_k"),
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapK,
                config.dim(),
                kvDim,
                false);
        matVec(
                layer,
                layerIndex,
                "attn_v_proj",
                "attn_v",
                require(weights.wvLayered, layerIndex, "attn_v"),
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapV,
                config.dim(),
                kvDim,
                false);

        layer.task(
                tn("attn_split_query_gate"),
                Qwen35AttentionKernels::splitQueryGate,
                context,
                qwen35State.workspace.wrapQ,
                qwen35State.workspace.wrapAttnQ,
                qwen35State.workspace.wrapAttnGate,
                config.numberOfHeads(),
                headDim);

        layer.task(
                tn("attn_qk_norm"),
                Qwen3Kernels::fusedQKRmsNorm,
                context,
                qwen35State.workspace.wrapAttnQ,
                qwen35State.workspace.wrapK,
                require(weights.attnQNorm, layerIndex, "attn_q_norm").asFloatArray(),
                require(weights.attnKNorm, layerIndex, "attn_k_norm").asFloatArray(),
                config.numberOfHeads(),
                config.numberOfKeyValueHeads(),
                headDim,
                headDim,
                config.rmsNormEps());

        layer.task(
                tn("attn_rope"),
                Qwen35AttentionKernels::ropeNeoxPartial,
                context,
                qwen35State.workspace.positionHolder,
                qwen35State.workspace.wrapAttnQ,
                qwen35State.workspace.wrapK,
                weights.freq_cis_realFlat.asFloatArray(),
                weights.freq_cis_imagFlat.asFloatArray(),
                config.numberOfHeads(),
                config.numberOfKeyValueHeads(),
                headDim,
                config.ropeDimensionCount());

        // The key/value store is sized by the blocks that attend, so this layer addresses it by
        // its dense index. Its own index would run four times past the end of the store.
        int kvLayer = config.keyValueLayerIndex(layerIndex);
        if (fp16Kv()) {
            layer.task(
                    tn("attn_kv_append"),
                    Qwen35AttentionKernels::appendKeyValueFP16Paged,
                    context,
                    qwen35State.workspace.positionHolder,
                    qwen35State.workspace.wrapK,
                    qwen35State.workspace.wrapV,
                    qwen35State.workspace.wrapKeyCacheFP16,
                    qwen35State.workspace.wrapValueCacheFP16,
                    qwen35State.workspace.wrapBlockTable,
                    kvDim,
                    kvLayer,
                    qwen35State.kvBlockCfg,
                    qwen35State.kvBlockStride);
        } else {
            layer.task(
                    tn("attn_kv_append"),
                    Qwen35AttentionKernels::appendKeyValuePaged,
                    context,
                    qwen35State.workspace.positionHolder,
                    qwen35State.workspace.wrapK,
                    qwen35State.workspace.wrapV,
                    qwen35State.workspace.wrapKeyCache,
                    qwen35State.workspace.wrapValueCache,
                    qwen35State.workspace.wrapBlockTable,
                    kvDim,
                    kvLayer,
                    qwen35State.kvBlockCfg,
                    qwen35State.kvBlockStride);
        }

        // @formatter:off
        // Split-KV decode attention, a warp per (head, split), where attentionSplits() admits it
        // (FP16 store, a 256-wide head, CUDA); the single-workgroup online-softmax kernel
        // otherwise. The single-workgroup kernel sizes its shared memory from the head width; the
        // shared 128-wide split-KV kernel of the other families cannot take this family's head.
        // @formatter:on
        int splits = attentionSplits();
        if (splits > 1) {
            // Phase 1: each head's key/value range split across `splits` workgroups, partials into
            // wrapAttSplit in the compact layout the combine expects.
            layer.task(
                    tn("attention"),
                    TransformerPagedKvKernels::processHeadsFlashAttentionSplitKVFP16PagedWarp,
                    context,
                    qwen35State.workspace.wrapAttnQ,
                    qwen35State.workspace.wrapKeyCacheFP16,
                    qwen35State.workspace.wrapValueCacheFP16,
                    qwen35State.workspace.wrapAttSplit,
                    config.numberOfHeads(),
                    headDim,
                    kvDim,
                    config.kvMul(),
                    qwen35State.workspace.positionHolder,
                    kvLayer,
                    qwen35State.workspace.wrapBlockTable,
                    qwen35State.kvBlockCfg,
                    qwen35State.kvBlockStride,
                    splits);
            // Phase 2: merge the per-head partials into wrapXb, where the per-head kernel would
            // have written directly.
            layer.task(
                    tn("attention_combine"),
                    TransformerComputeKernelsLayered::combineSplitKVAttention,
                    context,
                    qwen35State.workspace.wrapAttSplit,
                    qwen35State.workspace.wrapXb,
                    config.numberOfHeads(),
                    headDim,
                    splits);
        } else if (fp16Kv()) {
            layer.task(
                    tn("attention"),
                    TransformerPagedKvKernels::processHeadsFlashAttentionFP16Paged,
                    context,
                    qwen35State.workspace.wrapAttnQ,
                    qwen35State.workspace.wrapKeyCacheFP16,
                    qwen35State.workspace.wrapValueCacheFP16,
                    qwen35State.workspace.wrapXb,
                    config.numberOfHeads(),
                    headDim,
                    kvDim,
                    config.kvMul(),
                    qwen35State.workspace.positionHolder,
                    kvLayer,
                    qwen35State.workspace.wrapBlockTable,
                    qwen35State.kvBlockCfg,
                    qwen35State.kvBlockStride);
        } else {
            layer.task(
                    tn("attention"),
                    TransformerPagedKvKernels::processHeadsFlashAttentionPaged,
                    context,
                    qwen35State.workspace.wrapAttnQ,
                    qwen35State.workspace.wrapKeyCache,
                    qwen35State.workspace.wrapValueCache,
                    qwen35State.workspace.wrapXb,
                    config.numberOfHeads(),
                    headDim,
                    kvDim,
                    config.kvMul(),
                    qwen35State.workspace.positionHolder,
                    kvLayer,
                    qwen35State.workspace.wrapBlockTable,
                    qwen35State.kvBlockCfg,
                    qwen35State.kvBlockStride);
        }

        // A logistic, not a SiLU: reusing the SwiGLU kernel would multiply by the gate twice.
        // The gated attention output lands in wrapXb, over the activation that was quantized.
        normedActivationQuantized = false;
        layer.task(
                tn("attn_output_gate"),
                Qwen35AttentionKernels::applyOutputGate,
                context,
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapAttnGate,
                attnDim);

        matVec(
                layer,
                layerIndex,
                "attn_output_proj",
                "attn_output",
                require(weights.woLayered, layerIndex, "attn_output"),
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapX,
                attnDim,
                config.dim(),
                true);
    }

    // @formatter:off
    /**
     * A Gated Delta Net block, from the normalized {@code wrapXb} back into {@code wrapX}.
     *
     * <p>Follows the host branch operation for operation, including the two orderings that are not
     * interchangeable: the fused projection is convolved <b>before</b> it is split, and the queries
     * and keys are normalized <b>after</b> the convolution.
     *
     * <p>The convolution window and the delta-net state are per-layer slices of one array each, so
     * a layer passes its offset rather than binding its own buffer — 48 buffers per kind would be
     * 48 transfers to arrange and keep resident.
     */
    // @formatter:on
    private void deltaNetBranch(TaskGraph layer, int layerIndex) {
        final int convDim = config.deltaNetConvDim();
        final int keyDim = config.deltaNetKeyDim();
        final int valueDim = config.deltaNetValueDim();
        final int valueHeads = config.numberOfValueHeads();
        final int headK = config.headKeyDim();
        final int headV = config.headValueDim();
        final int recurrent = config.recurrentLayerIndex(layerIndex);

        matVec(
                layer,
                layerIndex,
                "ssm_qkv_proj",
                "attn_qkv",
                require(weights.ssmQkv, layerIndex, "attn_qkv"),
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapSsmQkv,
                config.dim(),
                convDim,
                false);
        matVec(
                layer,
                layerIndex,
                "ssm_gate_proj",
                "attn_gate",
                require(weights.ssmGate, layerIndex, "attn_gate"),
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapSsmZ,
                config.dim(),
                valueDim,
                false);
        matVec(
                layer,
                layerIndex,
                "ssm_beta_proj",
                "ssm_beta",
                require(weights.ssmBeta, layerIndex, "ssm_beta"),
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapSsmBeta,
                config.dim(),
                valueHeads,
                false);
        matVec(
                layer,
                layerIndex,
                "ssm_alpha_proj",
                "ssm_alpha",
                require(weights.ssmAlpha, layerIndex, "ssm_alpha"),
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapSsmAlpha,
                config.dim(),
                valueHeads,
                false);

        layer.task(
                tn("ssm_decay_beta"),
                Qwen35DeltaNetKernels::decayAndBeta,
                context,
                qwen35State.workspace.wrapSsmAlpha,
                qwen35State.workspace.wrapSsmBeta,
                require(weights.ssmDtBias, layerIndex, "ssm_dt.bias").asFloatArray(),
                require(weights.ssmA, layerIndex, "ssm_a").asFloatArray(),
                valueHeads);

        layer.task(
                tn("ssm_conv"),
                Qwen35DeltaNetKernels::causalConv1d,
                context,
                qwen35State.workspace.wrapSsmQkv,
                require(weights.ssmConv1d, layerIndex, "ssm_conv1d").asFloatArray(),
                qwen35State.workspace.wrapConvState,
                qwen35State.workspace.wrapSsmConvOut,
                convDim,
                config.ssmConvKernel(),
                recurrent * config.convStateSize());
        layer.task(
                tn("ssm_conv_silu"),
                Qwen35DeltaNetKernels::siluInPlace,
                context,
                qwen35State.workspace.wrapSsmConvOut,
                convDim);

        layer.task(
                tn("ssm_split_qkv"),
                TransformerComputeKernels::splitThreeWay,
                context,
                qwen35State.workspace.wrapSsmConvOut,
                qwen35State.workspace.wrapSsmQ,
                qwen35State.workspace.wrapSsmK,
                qwen35State.workspace.wrapSsmV,
                keyDim,
                keyDim,
                valueDim);

        if (l2NormIsWide()) {
            // A workgroup per key head and a lane per element. Same equation, same epsilon
            // placement; the sum of squares becomes a shared tree, which is why this is not
            // bit-identical to the per-head lane.
            layer.task(
                    tn("ssm_l2norm_q"),
                    Qwen35DeltaNetKernels::l2NormPerHeadWide,
                    context,
                    qwen35State.workspace.wrapSsmQ,
                    headK,
                    config.rmsNormEps());
            layer.task(
                    tn("ssm_l2norm_k"),
                    Qwen35DeltaNetKernels::l2NormPerHeadWide,
                    context,
                    qwen35State.workspace.wrapSsmK,
                    headK,
                    config.rmsNormEps());
        } else {
            layer.task(
                    tn("ssm_l2norm_q"),
                    Qwen35DeltaNetKernels::l2NormPerHead,
                    context,
                    qwen35State.workspace.wrapSsmQ,
                    config.numberOfKeyHeads(),
                    headK,
                    config.rmsNormEps());
            layer.task(
                    tn("ssm_l2norm_k"),
                    Qwen35DeltaNetKernels::l2NormPerHead,
                    context,
                    qwen35State.workspace.wrapSsmK,
                    config.numberOfKeyHeads(),
                    headK,
                    config.rmsNormEps());
        }
        layer.task(
                tn("ssm_scale_q"),
                TransformerComputeKernels::scaleInPlace,
                context,
                qwen35State.workspace.wrapSsmQ,
                (float) (1.0 / Math.sqrt(headK)),
                keyDim);

        DeltaRuleGeometry geometry = deltaRuleGeometry();
        if (geometry == DeltaRuleGeometry.SPLIT8) {
            // Eight lanes a column, each taking sixteen rows. Same per-element arithmetic; the
            // two reductions are sums of eight folds, so this is not bit-identical to a one-lane
            // column.
            layer.task(
                    tn("ssm_delta_rule"),
                    Qwen35DeltaNetKernels::deltaRuleSplit8,
                    context,
                    qwen35State.workspace.wrapSsmQ,
                    qwen35State.workspace.wrapSsmK,
                    qwen35State.workspace.wrapSsmV,
                    qwen35State.workspace.wrapSsmAlpha,
                    qwen35State.workspace.wrapSsmBeta,
                    qwen35State.workspace.wrapDeltaState,
                    qwen35State.workspace.wrapSsmOut,
                    config.numberOfKeyHeads(),
                    headV,
                    recurrent * config.deltaNetStateSize());
        } else if (geometry == DeltaRuleGeometry.SPLIT2) {
            // Two lanes a column: the kernel the eight-part one replaced, for a device whose
            // workgroup limit admits 2 * headDim but not 8 * headDim.
            layer.task(
                    tn("ssm_delta_rule"),
                    Qwen35DeltaNetKernels::deltaRuleSplit,
                    context,
                    qwen35State.workspace.wrapSsmQ,
                    qwen35State.workspace.wrapSsmK,
                    qwen35State.workspace.wrapSsmV,
                    qwen35State.workspace.wrapSsmAlpha,
                    qwen35State.workspace.wrapSsmBeta,
                    qwen35State.workspace.wrapDeltaState,
                    qwen35State.workspace.wrapSsmOut,
                    config.numberOfKeyHeads(),
                    headV,
                    recurrent * config.deltaNetStateSize());
        } else {
            layer.task(
                    tn("ssm_delta_rule"),
                    Qwen35DeltaNetKernels::deltaRule,
                    context,
                    qwen35State.workspace.wrapSsmQ,
                    qwen35State.workspace.wrapSsmK,
                    qwen35State.workspace.wrapSsmV,
                    qwen35State.workspace.wrapSsmAlpha,
                    qwen35State.workspace.wrapSsmBeta,
                    qwen35State.workspace.wrapDeltaState,
                    qwen35State.workspace.wrapSsmOut,
                    valueHeads,
                    config.numberOfKeyHeads(),
                    headV,
                    recurrent * config.deltaNetStateSize());
        }

        if (gatedNormIsWide()) {
            // A workgroup per head and a lane per element. Same equation, same gate position,
            // same epsilon; the sum of squares becomes a shared tree, which is why this is not
            // bit-identical to the per-head lane below.
            layer.task(
                    tn("ssm_gated_norm"),
                    Qwen35DeltaNetKernels::gatedNormPerHeadWide,
                    context,
                    qwen35State.workspace.wrapSsmOut,
                    qwen35State.workspace.wrapSsmZ,
                    require(weights.ssmNorm, layerIndex, "ssm_norm").asFloatArray(),
                    headV,
                    config.rmsNormEps());
        } else {
            layer.task(
                    tn("ssm_gated_norm"),
                    Qwen35DeltaNetKernels::gatedNormPerHead,
                    context,
                    qwen35State.workspace.wrapSsmOut,
                    qwen35State.workspace.wrapSsmZ,
                    require(weights.ssmNorm, layerIndex, "ssm_norm").asFloatArray(),
                    valueHeads,
                    headV,
                    config.rmsNormEps());
        }

        ssmActivationQuantized = false;
        if (DP4A && packedScratchHolds(config.deltaNetValueDim())) {
            // The delta-net readout, quantized fresh. It is none of the activations quantized
            // elsewhere in this layer, and it reuses their scratch: every projection that read
            // those is behind us in this graph, and the arrays are sized for the feed-forward's
            // hidden width, which is wider than this.
            layer.task(
                    tn("ssm_out_quantize"),
                    TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                    context,
                    qwen35State.workspace.wrapSsmOut,
                    qwen35State.workspace.wrapXbQuants,
                    qwen35State.workspace.wrapXbScales,
                    qwen35State.workspace.wrapXbSums);
            // The scratch now describes the readout, so whatever the branch quantized from wrapXb
            // no longer holds -- and this is the point at which a later reader would be wrong.
            normedActivationQuantized = false;
            ssmActivationQuantized = true;
        }

        matVec(
                layer,
                layerIndex,
                "ssm_out_proj",
                "ssm_out",
                require(weights.ssmOut, layerIndex, "ssm_out"),
                qwen35State.workspace.wrapSsmOut,
                qwen35State.workspace.wrapX,
                valueDim,
                config.dim(),
                true);
    }

    // ── transfers ─────────────────────────────────────────────────────────────

    // @formatter:off
    /**
     * The graph that has already uploaded this layer's weights, or {@code null} to upload them
     * here.
     *
     * <p>A weight array bound with {@code transferToDevice} in two graphs of one execution plan
     * gets a device buffer in each, so a plan holding both a batch-prefill and a decode family
     * would hold the model twice. The decode family consumes what the batch family uploaded.
     */
    // @formatter:on
    protected String weightSourceGraphName(int layerIndex) {
        return null;
    }

    /**
     * This layer's weights, uploaded once in the graph that first reads them.
     *
     * <p>A weight array bound with {@code transferToDevice} in two graphs of one execution plan
     * gets a device buffer in each, so a layer uploads only its own and never another's.
     */
    private void transferLayerWeights(TaskGraph layer, int layerIndex) {
        List<Object> tensors = new ArrayList<>();
        tensors.add(weights.rms_att_weightLayered[layerIndex].asFloatArray());
        tensors.add(weights.rms_ffn_weightLayered[layerIndex].asFloatArray());
        tensors.add(deviceArray(require(weights.w1Layered, layerIndex, "ffn_gate")));
        tensors.add(deviceArray(require(weights.w2Layered, layerIndex, "ffn_down")));
        tensors.add(deviceArray(require(weights.w3Layered, layerIndex, "ffn_up")));
        if (config.isRecurrentLayer(layerIndex)) {
            tensors.add(deviceArray(require(weights.ssmQkv, layerIndex, "attn_qkv")));
            tensors.add(deviceArray(require(weights.ssmGate, layerIndex, "attn_gate")));
            tensors.add(deviceArray(require(weights.ssmAlpha, layerIndex, "ssm_alpha")));
            tensors.add(deviceArray(require(weights.ssmBeta, layerIndex, "ssm_beta")));
            tensors.add(deviceArray(require(weights.ssmOut, layerIndex, "ssm_out")));
            tensors.add(require(weights.ssmConv1d, layerIndex, "ssm_conv1d").asFloatArray());
            tensors.add(require(weights.ssmDtBias, layerIndex, "ssm_dt.bias").asFloatArray());
            tensors.add(require(weights.ssmA, layerIndex, "ssm_a").asFloatArray());
            tensors.add(require(weights.ssmNorm, layerIndex, "ssm_norm").asFloatArray());
        } else {
            tensors.add(deviceArray(require(weights.wqLayered, layerIndex, "attn_q")));
            tensors.add(deviceArray(require(weights.wkLayered, layerIndex, "attn_k")));
            tensors.add(deviceArray(require(weights.wvLayered, layerIndex, "attn_v")));
            tensors.add(deviceArray(require(weights.woLayered, layerIndex, "attn_output")));
            tensors.add(require(weights.attnQNorm, layerIndex, "attn_q_norm").asFloatArray());
            tensors.add(require(weights.attnKNorm, layerIndex, "attn_k_norm").asFloatArray());
        }
        String source = weightSourceGraphName(layerIndex);
        if (source != null) {
            layer.consumeFromDevice(source, tensors.toArray());
        } else {
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, tensors.toArray());
        }
    }

    /** A tensor's device array, in whatever representation it is retained in. */
    private static Object deviceArray(TornadoTensor tensor) {
        return switch (tensor.dataType()) {
            case F32 -> tensor.asFloatArray();
            case F16 -> tensor.asHalfFloatArray();
            default -> tensor.asByteArray();
        };
    }

    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex == 0) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    qwen35State.workspace.positionHolder,
                    qwen35State.workspace.temp,
                    qwen35State.workspace.tempFFN,
                    qwen35State.workspace.wrapBlockTable);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    qwen35State.workspace.wrapXb,
                    qwen35State.workspace.wrapQ,
                    qwen35State.workspace.wrapAttnQ,
                    qwen35State.workspace.wrapAttnGate,
                    qwen35State.workspace.wrapK,
                    qwen35State.workspace.wrapV,
                    keyStore(),
                    valueStore(),
                    qwen35State.workspace.wrapAtt,
                    qwen35State.workspace.wrapAttSplit,
                    qwen35State.workspace.wrapHb);
            if (DP4A) {
                layer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION,
                        qwen35State.workspace.wrapXbQuants,
                        qwen35State.workspace.wrapXbScales,
                        qwen35State.workspace.wrapXbSums);
            }
            // The recurrent state persists across tokens and is updated in place, so it is
            // uploaded once — zeroed — and never read back. Uploading it every execution would
            // overwrite the device's own history with the host's stale copy.
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    qwen35State.workspace.wrapSsmQkv,
                    qwen35State.workspace.wrapSsmConvOut,
                    qwen35State.workspace.wrapSsmZ,
                    qwen35State.workspace.wrapSsmAlpha,
                    qwen35State.workspace.wrapSsmBeta,
                    qwen35State.workspace.wrapSsmQ,
                    qwen35State.workspace.wrapSsmK,
                    qwen35State.workspace.wrapSsmV,
                    qwen35State.workspace.wrapSsmOut,
                    qwen35State.workspace.wrapConvState,
                    qwen35State.workspace.wrapDeltaState);
        } else {
            // The graph holding the previous layer, which is not "layer_{i-1}" when a family
            // puts several layers in one graph.
            String predecessor = layerGraphName(layerIndex - 1);
            layer.consumeFromDevice(
                    predecessor,
                    context,
                    qwen35State.workspace.wrapXb,
                    qwen35State.workspace.wrapQ,
                    qwen35State.workspace.wrapAttnQ,
                    qwen35State.workspace.wrapAttnGate,
                    qwen35State.workspace.wrapK,
                    qwen35State.workspace.wrapV,
                    keyStore(),
                    valueStore(),
                    qwen35State.workspace.wrapAtt,
                    qwen35State.workspace.wrapAttSplit,
                    qwen35State.workspace.wrapHb,
                    qwen35State.workspace.positionHolder);
            layer.consumeFromDevice(predecessor, qwen35State.workspace.wrapBlockTable);
            if (DP4A) {
                layer.consumeFromDevice(
                        predecessor,
                        qwen35State.workspace.wrapXbQuants,
                        qwen35State.workspace.wrapXbScales,
                        qwen35State.workspace.wrapXbSums);
            }
            layer.consumeFromDevice(
                    predecessor,
                    qwen35State.workspace.temp,
                    qwen35State.workspace.tempFFN,
                    qwen35State.workspace.wrapSsmQkv,
                    qwen35State.workspace.wrapSsmConvOut,
                    qwen35State.workspace.wrapSsmZ,
                    qwen35State.workspace.wrapSsmAlpha,
                    qwen35State.workspace.wrapSsmBeta,
                    qwen35State.workspace.wrapSsmQ,
                    qwen35State.workspace.wrapSsmK,
                    qwen35State.workspace.wrapSsmV,
                    qwen35State.workspace.wrapSsmOut,
                    qwen35State.workspace.wrapConvState,
                    qwen35State.workspace.wrapDeltaState);
        }
        return layer;
    }

    // ── worker grids ──────────────────────────────────────────────────────────

    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        WorkerGrid rmsReduce =
                rmsReduceWorker(
                        WorkerGridFactory.createRmsNormWorker(config.dim(), state.localSize));
        WorkerGrid rmsApply = WorkerGridFactory.createRmsNormWorker(config.dim(), state.localSize);
        WorkerGrid rmsFinalize =
                WorkerGridFactory.createRmsNormWorker(config.dim(), state.localSize);

        final int headDim = config.numberOfHeadsKey();
        WorkerGrid queryGate =
                WorkerGridFactory.genericWorker(config.queryGateDim(), ELEMENTWISE_LOCAL);
        WorkerGrid qkNorm =
                WorkerGridFactory.genericWorker(
                        (config.numberOfHeads() + config.numberOfKeyValueHeads()) * headDim,
                        headDim);
        WorkerGrid rope =
                WorkerGridFactory.genericWorker(
                        config.numberOfHeads() * (config.ropeDimensionCount() / 2), 32);
        WorkerGrid kvAppend = WorkerGridFactory.genericWorker(config.kvDim(), ELEMENTWISE_LOCAL);
        int splits = attentionSplits();
        // Split-KV launches nHeads*splits warps of SPLIT_KV_LOCAL lanes, followed by a combine
        // pass of one workgroup per head. With splits == 1 the per-head kernel keeps the worker it
        // has always had.
        WorkerGrid attention =
                splits > 1
                        ? WorkerGridFactory.genericWorker(
                                config.numberOfHeads() * splits * SPLIT_KV_LOCAL, SPLIT_KV_LOCAL)
                        : WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), headDim);
        WorkerGrid attentionCombine =
                WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), headDim);
        WorkerGrid outputGate =
                WorkerGridFactory.genericWorker(
                        config.attentionOutputInputDim(), ELEMENTWISE_LOCAL);

        WorkerGrid convDim =
                WorkerGridFactory.genericWorker(config.deltaNetConvDim(), ELEMENTWISE_LOCAL);
        WorkerGrid keyDim =
                WorkerGridFactory.genericWorker(config.deltaNetKeyDim(), ELEMENTWISE_LOCAL);
        WorkerGrid keyHeads =
                WorkerGridFactory.genericWorker(
                        config.numberOfKeyHeads(), config.numberOfKeyHeads());
        WorkerGrid valueHeads =
                WorkerGridFactory.genericWorker(
                        config.numberOfValueHeads(), config.numberOfValueHeads());
        // A workgroup per key head, a lane per element of it, on the same terms as the value
        // head's grid below.
        WorkerGrid l2NormWide =
                WorkerGridFactory.genericWorker(
                        config.numberOfKeyHeads() * config.headKeyDim(), config.headKeyDim());
        // A workgroup per value head, a lane per element of it, where the head's width allows
        // the shared tree; otherwise the one-lane-per-head grid below is what is registered.
        WorkerGrid gatedNormWide =
                WorkerGridFactory.genericWorker(
                        config.numberOfValueHeads() * config.headValueDim(), config.headValueDim());
        // The grid follows the same decision as the task: a workgroup of parts * headDim per
        // value head for a split kernel, the elementwise default for the lane-per-column one.
        WorkerGrid deltaRule = deltaRuleWorker(deltaRuleGeometry(), config);

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            // The same graph and the same task qualification the tasks were built with; a
            // grouped family puts two layers in one graph and distinguishes them by task prefix.
            String prefix = layerGraphName(layer) + "." + layerTaskPrefix(layer);
            scheduler.addWorkerGrid(prefix + "attn_rms_reduce", rmsReduce);
            scheduler.addWorkerGrid(prefix + "ffn_rms_reduce", rmsReduce);
            if (shouldUseFinalNormalization()) {
                scheduler.addWorkerGrid(prefix + "attn_rms_finalize", rmsFinalize);
                scheduler.addWorkerGrid(prefix + "ffn_rms_finalize", rmsFinalize);
            }
            // Where the apply carries the quantization it takes the quantization's grid: one
            // 32-lane workgroup per block, which is the ownership the block maximum needs.
            if (DP4A) {
                scheduler.addWorkerGrid(
                        prefix + "attn_rms_apply",
                        WorkerGridFactory.genericWorker(config.dim(), 32));
                scheduler.addWorkerGrid(
                        prefix + "ffn_rms_apply",
                        WorkerGridFactory.genericWorker(config.dim(), 32));
                scheduler.addWorkerGrid(
                        prefix + "ffn_down_quantize",
                        WorkerGridFactory.genericWorker(config.hiddenDim(), 32));
            } else {
                scheduler.addWorkerGrid(prefix + "attn_rms_apply", rmsApply);
                scheduler.addWorkerGrid(prefix + "ffn_rms_apply", rmsApply);
            }
            scheduler.addWorkerGrid(prefix + "ffn_gate_up", matVecWorker(config.hiddenDim()));
            scheduler.addWorkerGrid(prefix + "ffn_down_proj", matVecWorker(config.dim()));

            if (config.isRecurrentLayer(layer)) {
                scheduler.addWorkerGrid(
                        prefix + "ssm_qkv_proj", matVecWorker(config.deltaNetConvDim()));
                scheduler.addWorkerGrid(
                        prefix + "ssm_gate_proj", matVecWorker(config.deltaNetValueDim()));
                if (DP4A && packedScratchHolds(config.deltaNetValueDim())) {
                    scheduler.addWorkerGrid(
                            prefix + "ssm_out_quantize",
                            WorkerGridFactory.genericWorker(config.deltaNetValueDim(), 32));
                }
                scheduler.addWorkerGrid(
                        prefix + "ssm_beta_proj", matVecWorker(config.numberOfValueHeads()));
                scheduler.addWorkerGrid(
                        prefix + "ssm_alpha_proj", matVecWorker(config.numberOfValueHeads()));
                scheduler.addWorkerGrid(prefix + "ssm_decay_beta", valueHeads);
                scheduler.addWorkerGrid(prefix + "ssm_conv", convDim);
                scheduler.addWorkerGrid(prefix + "ssm_conv_silu", convDim);
                scheduler.addWorkerGrid(prefix + "ssm_split_qkv", convDim);
                scheduler.addWorkerGrid(
                        prefix + "ssm_l2norm_q", l2NormIsWide() ? l2NormWide : keyHeads);
                scheduler.addWorkerGrid(
                        prefix + "ssm_l2norm_k", l2NormIsWide() ? l2NormWide : keyHeads);
                scheduler.addWorkerGrid(prefix + "ssm_scale_q", keyDim);
                scheduler.addWorkerGrid(prefix + "ssm_delta_rule", deltaRule);
                scheduler.addWorkerGrid(
                        prefix + "ssm_gated_norm", gatedNormIsWide() ? gatedNormWide : valueHeads);
                scheduler.addWorkerGrid(prefix + "ssm_out_proj", matVecWorker(config.dim()));
            } else {
                scheduler.addWorkerGrid(
                        prefix + "attn_q_proj", matVecWorker(config.queryGateDim()));
                scheduler.addWorkerGrid(prefix + "attn_k_proj", matVecWorker(config.kvDim()));
                scheduler.addWorkerGrid(prefix + "attn_v_proj", matVecWorker(config.kvDim()));
                scheduler.addWorkerGrid(prefix + "attn_split_query_gate", queryGate);
                scheduler.addWorkerGrid(prefix + "attn_qk_norm", qkNorm);
                scheduler.addWorkerGrid(prefix + "attn_rope", rope);
                scheduler.addWorkerGrid(prefix + "attn_kv_append", kvAppend);
                scheduler.addWorkerGrid(prefix + "attention", attention);
                if (splits > 1) {
                    scheduler.addWorkerGrid(prefix + "attention_combine", attentionCombine);
                }
                scheduler.addWorkerGrid(prefix + "attn_output_gate", outputGate);
                scheduler.addWorkerGrid(prefix + "attn_output_proj", matVecWorker(config.dim()));
            }
        }
        return scheduler;
    }

    /** One workgroup per output row, which is how every matrix-vector kernel here is written. */
    private static WorkerGrid matVecWorker(int rows) {
        return WorkerGridFactory.genericWorker(rows * MATVEC_LOCAL, MATVEC_LOCAL);
    }

    /** The delta-rule worker grid for a geometry: one workgroup per value head. */
    static WorkerGrid deltaRuleWorker(DeltaRuleGeometry geometry, Qwen35Configuration config) {
        int headDim = config.headValueDim();
        if (geometry == DeltaRuleGeometry.LANE_PER_COLUMN) {
            return WorkerGridFactory.genericWorker(
                    config.numberOfValueHeads() * headDim, ELEMENTWISE_LOCAL);
        }
        return WorkerGridFactory.genericWorker(
                config.numberOfValueHeads() * geometry.localSize(headDim),
                geometry.localSize(headDim));
    }
}
