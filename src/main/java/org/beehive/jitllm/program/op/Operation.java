package org.beehive.jitllm.program.op;

import java.util.List;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * One unit of work in a forward pass, described independently of backend and model family.
 *
 * <p>Sealed against the closed {@link OperationKind} set: adding a model family assembles these
 * differently, it does not add to them.
 */
public sealed interface Operation
        permits RmsNorm,
                RoPE,
                MatVec,
                MatMul,
                KvAppend,
                Attention,
                Softmax,
                SwiGLU,
                GeGLU,
                ResidualAdd,
                BiasAdd,
                Scale,
                SplitFusedQkv,
                SplitGateUp,
                EmbeddingLookup,
                VocabProjection,
                LogitSoftCap,
                MoeRouter,
                ExpertFeedForward,
                WeightedAccumulate,
                ArgMax,
                Sample,
                L2Norm,
                CausalConv1d,
                DeltaRuleUpdate,
                GatedNorm {

    /** Which operation this is. Every implementation returns one fixed kind. */
    OperationKind kind();

    /**
     * The representation that selects this operation's implementation.
     *
     * <p>For an operation over model weights that is <b>the representation the weights were
     * materialized in</b> — not what the file held. It is the same axis {@code ForwardPlanFactory}
     * already dispatches on first, and it is per operation rather than per program because a
     * K-quant model is mixed: its embeddings and output are typically {@code Q6_K} or {@code F32}
     * while its attention weights are {@code Q4_K}. For an operation over activations only, it is
     * the representation of those activations.
     *
     * <p>Parameterization lives here and at dispatch, <b>not inside kernel bodies</b>: TornadoVM
     * compiles per concrete native array type and Java has no generics over primitives, so one
     * kernel body cannot serve every representation.
     *
     * <p>Whether a target can actually execute this operation at this representation is {@link
     * OperationSupport}'s question, and it is answered before invocation.
     */
    DataType dataType();

    /**
     * The operands this operation reads, in a stable order.
     *
     * <p>Stable because an operation is part of a program signature and therefore part of a cache
     * key: a list whose order depended on construction would make two identical programs compare
     * unequal.
     */
    List<OperandRef> inputs();

    /**
     * The operands this operation writes, in a stable order.
     *
     * <p>An operand may appear in both {@link #inputs()} and here. That is not a defect: rotary
     * position embedding rewrites the query and key projections in place, and saying so is more
     * honest than inventing a distinct output name for the same storage.
     */
    List<OperandRef> outputs();
}
