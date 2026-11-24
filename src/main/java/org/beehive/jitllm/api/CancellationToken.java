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
 * Asks a generation to stop early.
 *
 * <p>Attach a token to a request with {@link GenerationRequest.Builder#cancellation}, and call
 * {@link #cancel()} from <b>any thread</b> while {@link GenerationSession#generate} runs — a server
 * whose client disconnected, a UI's stop button, a coroutine being cancelled. Generation stops
 * after the token being produced, the session stays usable, and the result carries the text
 * produced so far with {@link FinishReason#CANCELLED}.
 *
 * <p>A token is one-shot: once cancelled it stays cancelled, and a request built with a cancelled
 * token returns immediately without generating. Use a new token per request.
 */
@Experimental
public final class CancellationToken {

    private volatile boolean cancelled;

    /** Asks the generation using this token to stop. Idempotent; safe from any thread. */
    public void cancel() {
        cancelled = true;
    }

    /** Whether {@link #cancel()} has been called. */
    public boolean isCancelled() {
        return cancelled;
    }
}
