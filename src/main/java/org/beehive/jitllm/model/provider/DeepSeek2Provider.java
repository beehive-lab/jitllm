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
import org.beehive.jitllm.model.loader.DeepSeek2ModelLoader;

/** The {@code deepseek2} family: latent attention with a mixture of experts (GLM-4.7-Flash). */
public final class DeepSeek2Provider extends FamilyProvider {

    public DeepSeek2Provider() {
        super("deepseek2");
    }

    @Override
    protected DeepSeek2ModelLoader loader(
            ModelSource source, int contextLength, boolean useTornadovm) {
        return new DeepSeek2ModelLoader(
                source.gguf().getFileChannel(), source.gguf(), contextLength, useTornadovm);
    }
}
