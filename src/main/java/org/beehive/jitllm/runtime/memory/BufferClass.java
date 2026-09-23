package org.beehive.jitllm.runtime.memory;

import org.beehive.jitllm.api.Experimental;

/**
 * What a device-resident buffer is <b>for</b> — the classification a memory plan reasons over.
 *
 * <p>Neutral by construction: these are logical roles, not TornadoVM concepts. A backend decides
 * how many physical allocations a role costs; this layer only says what exists and who owns it.
 *
 * <p><b>Why the per-layer / global split exists.</b> It is the difference that decides batched
 * prefill's footprint. Batched prefill builds a second family of layer graphs, so the buffers a
 * *layer* graph binds are allocated twice while the buffers bound by the single activation and
 * logits graphs are allocated once. Measured on Llama-3.2-1B-F16, batched prefill costs 1872 MiB
 * more than single-token, and the per-layer weights are 1856 MiB — the *whole* weight set is 2357
 * MiB, which would not have matched. Collapsing the two into one "weights" class would have made
 * that prediction wrong by 500 MiB.
 */
@Experimental
public enum BufferClass {

    /** Weights a layer graph binds: attention and FFN projections, per-layer norms. */
    WEIGHTS_PER_LAYER,

    /** Weights bound outside the layer graphs: embeddings, the output projection, RoPE tables. */
    WEIGHTS_GLOBAL,

    /** Key/value cache storage, sized by context capacity and KV representation. */
    KV_CACHE,

    /** Fixed per-session activation and attention scratch. */
    ACTIVATION_WORKSPACE,

    /** Staging that exists only when a batched prefill capacity is configured. */
    BATCH_STAGING,

    /** Position, block-table and control carriers. */
    CONTROL,

    /** Result and sampling carriers. */
    RESULT;

    /**
     * Whether this class is bound by the per-layer graphs, and so is subject to layer-family reuse.
     */
    public boolean isPerLayer() {
        return this == WEIGHTS_PER_LAYER;
    }
}
