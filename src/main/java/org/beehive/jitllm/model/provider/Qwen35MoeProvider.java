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
package org.beehive.jitllm.model.provider;

import org.beehive.jitllm.format.ModelSource;
import org.beehive.jitllm.model.loader.Qwen35MoeModelLoader;

/**
 * The {@code qwen35moe} architecture: the {@code qwen35} hybrid stack with a mixture-of-experts
 * feed-forward (Qwen3.6-35B-A3B). The same loader reads it; the configuration it builds carries the
 * experts.
 */
public final class Qwen35MoeProvider extends FamilyProvider {

    public Qwen35MoeProvider() {
        super("qwen35moe");
    }

    @Override
    protected Qwen35MoeModelLoader loader(
            ModelSource source, int contextLength, boolean useTornadovm) {
        return new Qwen35MoeModelLoader(
                source.gguf().getFileChannel(), source.gguf(), contextLength, useTornadovm);
    }
}
