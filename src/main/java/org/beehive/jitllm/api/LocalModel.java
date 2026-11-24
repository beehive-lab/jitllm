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
 * A loaded model: weights, configuration and whatever the backend compiled for them.
 *
 * <p><b>Thread-safe.</b> Immutable after load; any number of threads may read it and open sessions
 * from it (through a capability such as {@link TextGenerationModel}).
 *
 * <p><b>Generation is not assumed.</b> A model that only produces embeddings is a {@code
 * LocalModel} and nothing more; generation lives on {@link TextGenerationModel}, which is where
 * {@code newSession()} is declared. There is no {@code forward(.)}, no sampler and no plan accessor
 * here: execution belongs below the model, generation policy above it.
 *
 * <h2>Closing</h2>
 *
 * <p>{@link #close()} <b>throws {@link IllegalStateException} while sessions from this model are
 * still open</b>, naming them, and never force-closes them — closing another thread's session
 * mid-generation is a worse failure than an exception. A failed close has no effect: the model
 * stays open and usable, so the caller can close the sessions and try again. A successful close is
 * idempotent, and no session may be opened afterwards.
 *
 * <p>The natural spelling is nested try-with-resources, model outer and session inner.
 */
public interface LocalModel extends AutoCloseable {

    /** Identity and context length. */
    @Experimental
    ModelInfo info();

    /** The network's shape, read-only. */
    @Experimental
    ModelConfiguration configuration();

    /**
     * Releases the weights and everything compiled for them.
     *
     * @throws IllegalStateException if any session created from this model is still open; the model
     *     remains open and no resource is released
     */
    @Override
    void close();
}
