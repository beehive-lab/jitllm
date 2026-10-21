package org.beehive.jitllm.backend.tornado;

/**
 * A plan that prefills a prompt in chunks of several tokens and then decodes one token at a time:
 * {@link TornadoVMMasterPlanBatchPrefillDecode} on one device, or a plan split across several.
 *
 * <p>The host stages each chunk in the session state ({@code TornadoBatchPrefillPass}): its
 * embeddings and {@code batchStartPosHolder} (start position, active rows, KV slot). The plan then
 * runs the chunk through every layer and keeps the key/value cache on the device for the decode
 * steps that follow. A prefill produces no logits: the first decode step does.
 */
public interface BatchPrefillDecodePlan extends TornadoVMMasterPlan {

    /** Runs the staged chunk through the batched prefill graphs. */
    void tornadoVMForwardBatchPrefill();

    /**
     * Whether chunks after the first take a separate family of prefill graphs. Only a plan whose
     * first chunk uses a kernel that cannot continue a sequence has one.
     */
    default boolean hasBatchPrefillFallback() {
        return false;
    }

    /** Runs the staged chunk through the fallback prefill graphs. */
    default void tornadoVMForwardBatchPrefillFallback() {
        throw new UnsupportedOperationException("this plan has no fallback prefill graphs");
    }

    /**
     * Whether this plan runs a prompt's chunks overlapped, a later one starting before an earlier
     * one has finished: a pipeline, whose first device takes the next chunk while the next device
     * still works on the previous one.
     */
    default boolean overlapsPrefillChunks() {
        return false;
    }

    /**
     * Runs {@code chunks} prefill chunks in order, overlapped where {@link #overlapsPrefillChunks}.
     * {@code stageChunk.accept(c)} stages chunk {@code c} in the session state, as for {@link
     * #tornadoVMForwardBatchPrefill}; it is called in chunk order, each call after the previous
     * chunk's staging has been consumed, and possibly on another thread.
     */
    default void tornadoVMForwardBatchPrefillChunks(
            int chunks, java.util.function.IntConsumer stageChunk) {
        for (int c = 0; c < chunks; c++) {
            stageChunk.accept(c);
            tornadoVMForwardBatchPrefill();
        }
    }
}
