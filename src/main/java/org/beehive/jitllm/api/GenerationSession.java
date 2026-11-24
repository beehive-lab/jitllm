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
package org.beehive.jitllm.api;

/**
 * One sequence: a conversation, a completion, one thread's worth of work.
 *
 * <p><b>Not thread-safe.</b> A session carries the sequence's position and its key/value state; two
 * threads generating into one session would interleave a single conversation. Run several sessions
 * instead — they share the model's weights and cost only their own context.
 *
 * <p>The session holds its key/value state as a lease rather than owning storage, so blocks can be
 * shared between sequences and can outlive an individual session.
 *
 * <p><b>Multi-turn is a session that stays open.</b> Each {@code generate} continues from the
 * position the last one reached; {@link #reset()} starts the conversation over without paying for a
 * new session.
 *
 * <p>{@link #close()} is idempotent. Closing the model with a session still open is an error — see
 * {@link LocalModel#close()}.
 */
public interface GenerationSession extends AutoCloseable {

    /**
     * Generates a completion, continuing this session's sequence.
     *
     * @throws IllegalStateException if the session or its model is closed
     */
    GenerationResult generate(GenerationRequest request);

    /**
     * Prepares execution before the first request and returns the selected settings. Repeated calls
     * reuse the prepared plan and do not advance or reset the sequence. Implementations without
     * diagnostics may throw {@link UnsupportedOperationException}.
     */
    @Experimental
    default org.beehive.jitllm.runtime.backend.ExecutionInfo prepare() {
        throw new UnsupportedOperationException(
                "This session does not expose execution diagnostics");
    }

    /** How much of the context this sequence has consumed, in tokens. */
    int position();

    /** Discards the sequence and starts from an empty context. The session stays usable. */
    void reset();

    /** Releases the sequence's state. Idempotent. */
    @Override
    void close();
}
