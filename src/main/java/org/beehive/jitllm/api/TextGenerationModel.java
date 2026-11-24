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
 * The capability of generating text — held by models that can, absent on models that cannot.
 *
 * <p><b>Thread-safe</b>, like every {@link LocalModel}. The sessions it hands out are not: a
 * session is one sequence, driven by one thread at a time. Opening several sessions is how a caller
 * runs several conversations against one model, and the weights are shared, not copied.
 *
 * <p>Generation is a capability rather than a method on {@link LocalModel} so that a model which
 * cannot generate is never forced to implement a method it can only fail.
 */
public interface TextGenerationModel extends LocalModel {

    /** Opens a sequence with the model's own defaults. The caller owns and must close it. */
    GenerationSession newSession();

    /**
     * Opens a sequence with the given options.
     *
     * @throws IllegalStateException if the model has been closed
     * @throws IllegalArgumentException if the requested context length exceeds the model's
     */
    GenerationSession newSession(SessionOptions options);
}
