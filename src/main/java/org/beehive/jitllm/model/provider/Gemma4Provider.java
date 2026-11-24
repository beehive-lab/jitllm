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
import org.beehive.jitllm.model.loader.Gemma4ModelLoader;

/**
 * Gemma-4. Declares {@code gemma4} in {@code general.architecture} (verified against
 * gemma-4-E2B-it-Q8_0.gguf, which also names itself "Gemma-4-E2B-It"), and its weight set —
 * per-layer token embeddings, dual RoPE tables for sliding-window and full attention — is its own,
 * so no other loader can read it.
 */
public final class Gemma4Provider extends FamilyProvider {
    public Gemma4Provider() {
        super("gemma4");
    }

    @Override
    protected Gemma4ModelLoader loader(
            ModelSource source, int contextLength, boolean useTornadovm) {
        return new Gemma4ModelLoader(
                source.gguf().getFileChannel(), source.gguf(), contextLength, useTornadovm);
    }
}
