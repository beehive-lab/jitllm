/*
 * Copyright (c) 2024-2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
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
     * Returns once the device has finished the prefill just run; call it before anything else runs
     * on the plan. On Metal {@code execute()} returns with the command buffers still queued, so a
     * pp512 timer stopped without this misses the end of the prefill. A decode step waits anyway,
     * when it reads its logits; CUDA's {@code execute()} already waits.
     */
    default void awaitBatchPrefill() {}

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
     * The rows of the plan's narrow prefill graphs, or 0 when it has none: a chunk with no more
     * tokens than this runs on them, whatever its start position.
     */
    default int narrowPrefillRows() {
        return 0;
    }

    /** Runs the staged chunk through the narrow prefill graphs. */
    default void tornadoVMForwardBatchPrefillNarrow() {
        throw new UnsupportedOperationException("this plan has no narrow prefill graphs");
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
