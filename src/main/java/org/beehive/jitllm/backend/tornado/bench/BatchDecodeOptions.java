package org.beehive.jitllm.backend.tornado.bench;

/**
 * What a batched-decode run is configured with.
 *
 * <p>Eleven knobs that were eleven {@code -Dbatch.decode.*} system properties read from the middle
 * of a 400-line method. As properties they had three problems: two runs in one JVM could not
 * differ, a caller could not see what a run was configured with, and the coupling to {@code
 * jitllm.prefillBatchSize} was enforced by an exception thrown after the model had loaded.
 *
 * @param batchSize slots decoded together, {@code B}
 * @param decodeContext context length each slot is given
 * @param decodeTokens tokens decoded per slot
 * @param cudaGraphs whether the plan is captured as a CUDA graph
 * @param paged whether key/value storage is block-addressed
 * @param blockSize tokens per block, when paged
 * @param blocks blocks in the pool, when paged
 * @param continuous whether finished slots are refilled while others run
 * @param requests total requests to serve in continuous mode
 * @param minDecodeTokens the shortest a refilled request may be
 * @param prefixCache whether a shared prompt prefix is cached across requests, paged only
 * @param temperature sampling temperature; {@code 0} is greedy
 * @param deviceSample whether the argmax runs on the device, greedy only
 */
public record BatchDecodeOptions(
        int batchSize,
        int decodeContext,
        int decodeTokens,
        boolean cudaGraphs,
        boolean paged,
        int blockSize,
        int blocks,
        boolean continuous,
        int requests,
        int minDecodeTokens,
        boolean prefixCache,
        float temperature,
        boolean deviceSample) {

    public BatchDecodeOptions {
        if (batchSize < 1) {
            throw new IllegalArgumentException("batchSize must be at least 1: " + batchSize);
        }
        if (decodeTokens < 1) {
            throw new IllegalArgumentException("decodeTokens must be at least 1: " + decodeTokens);
        }
        if (prefixCache && !paged) {
            // It was silently ignored when unpaged: `paged && prefixCache`, with no way to tell a
            // run that ignored the setting from one that never had it.
            throw new IllegalArgumentException(
                    "prefixCache needs paged key/value storage; set paged, or clear prefixCache");
        }
        if (deviceSample && temperature != 0.0f) {
            throw new IllegalArgumentException(
                    "device sampling is greedy only; temperature was " + temperature);
        }
    }

    /** The defaults, for a batch of {@code B}. */
    public static BatchDecodeOptions of(int batchSize) {
        return new BatchDecodeOptions(
                batchSize,
                512,
                64,
                true,
                false,
                16,
                0,
                false,
                4 * batchSize,
                32,
                false,
                0.0f,
                true);
    }

    /** Blocks in the pool, defaulted from the batch and context when not set explicitly. */
    public int resolvedBlocks(int maxBlocksPerSlot) {
        return blocks > 0 ? blocks : batchSize * maxBlocksPerSlot;
    }
}
