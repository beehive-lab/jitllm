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

import org.beehive.jitllm.model.Configuration;

/** The internal {@link Configuration}, narrowed to what a user may see. */
final class ConfigurationView implements ModelConfiguration {

    private final Configuration delegate;

    ConfigurationView(Configuration delegate) {
        this.delegate = delegate;
    }

    @Override
    public int dimension() {
        return delegate.dim();
    }

    @Override
    public int hiddenDimension() {
        return delegate.hiddenDim();
    }

    @Override
    public int layers() {
        return delegate.numberOfLayers();
    }

    @Override
    public int attentionHeads() {
        return delegate.numberOfHeads();
    }

    @Override
    public int keyValueHeads() {
        return delegate.numberOfKeyValueHeads();
    }

    @Override
    public int vocabularySize() {
        return delegate.vocabularySize();
    }

    @Override
    public int maxContextLength() {
        return delegate.contextLengthModel();
    }
}
