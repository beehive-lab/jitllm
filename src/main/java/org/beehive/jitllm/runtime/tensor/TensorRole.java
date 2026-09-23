package org.beehive.jllm.runtime.tensor;

/**
 * What a tensor is for.
 *
 * <p>A <b>closed set owned by the runtime layer</b>, deliberately not a free-form string: a string
 * role becomes a second dispatch key that architecture-specific code can branch on, which is the
 * thing [Rule 15](./././././././docs/architecture/architecture.md) exists to prevent. Adding an
 * architecture must not mean adding a role — it means reusing these.
 *
 * <p>Seeded from the weight fields the loaders already read. Whether {@code Weights} itself becomes
 * a map keyed by role is a separate, still-open question (open question 6).
 */
public enum TensorRole {

    /** Token embedding table. */
    TOKEN_EMBEDDING,

    /** RMSNorm weights before attention. */
    ATTENTION_NORM,

    /** Query, key and value projections, and the fused QKV some architectures ship instead. */
    ATTENTION_QUERY,
    ATTENTION_KEY,
    ATTENTION_VALUE,
    ATTENTION_QKV,

    /** Per-head query and key norms (Qwen3). */
    ATTENTION_QUERY_NORM,
    ATTENTION_KEY_NORM,

    /** Learned bias vectors added to the query, key and value projections. */
    ATTENTION_QUERY_BIAS,
    ATTENTION_KEY_BIAS,
    ATTENTION_VALUE_BIAS,

    /** Attention output projection. */
    ATTENTION_OUTPUT,

    /** RMSNorm weights before the feed-forward block. */
    FFN_NORM,

    /** Feed-forward projections. */
    FFN_GATE,
    FFN_UP,

    /** A single matrix producing the gate and up projections end to end. */
    FFN_GATE_UP,

    FFN_DOWN,

    /** Final norm before the logits projection. */
    OUTPUT_NORM,

    /** Logits projection — often the embedding table again, tied. */
    OUTPUT,

    /** Precomputed rotary frequencies; not read from the file. */
    ROPE_FREQUENCIES,

    /**
     * Mixture-of-experts weights.
     *
     * <p>The three expert projections are <b>stacked</b> tensors holding every expert, not one
     * expert each: the expert index is arithmetic inside the operation, which is what keeps slicing
     * out of the descriptor vocabulary (and 's refusal of views). The shared expert is the
     * always-on one Qwen2-MoE runs alongside the routed ones, gated by a scalar.
     */
    MOE_ROUTER_GATE,
    MOE_EXPERT_GATE,
    MOE_EXPERT_UP,
    MOE_EXPERT_DOWN,
    MOE_SHARED_GATE,
    MOE_SHARED_UP,
    MOE_SHARED_DOWN,
    MOE_SHARED_GATE_INPUT,

    /** A tensor whose role this layer does not model — biases, and anything new. */
    OTHER
}
