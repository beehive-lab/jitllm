package org.beehive.jllm.api;

/**
 * The model's hyperparameters, read-only.
 *
 * <p>Immutable and thread-safe. This is a narrowed view: it carries the shape of the network, not
 * the internal configuration record, so that the internal type stays free to change and no
 * format-level or backend-level value reaches a user.
 */
@Experimental
public interface ModelConfiguration {

    /** Embedding dimension. */
    int dimension();

    /** Feed-forward hidden dimension. */
    int hiddenDimension();

    int layers();

    int attentionHeads();

    /** Key/value heads — fewer than {@link #attentionHeads()} under grouped-query attention. */
    int keyValueHeads();

    int vocabularySize();

    /** The model's own maximum sequence length, which may exceed the loaded context length. */
    int maxContextLength();
}
