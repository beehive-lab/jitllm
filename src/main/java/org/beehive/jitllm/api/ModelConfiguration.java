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
