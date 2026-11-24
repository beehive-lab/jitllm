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
import org.beehive.jitllm.model.loader.Qwen35ModelLoader;

/**
 * The {@code qwen35} architecture: Qwen3.5, Qwen3.6 and Qwen3.8, which share one hybrid stack.
 *
 * <p>Recognition needs no case in {@link GgufRecognition}. These files declare {@code
 * general.architecture = qwen35} and nothing else claims that name, so the default branch — an
 * architecture that simply means itself — already resolves it. That is what adding a family with
 * one class and one service line is supposed to look like.
 */
public final class Qwen35Provider extends FamilyProvider {

    public Qwen35Provider() {
        super("qwen35");
    }

    @Override
    protected Qwen35ModelLoader loader(
            ModelSource source, int contextLength, boolean useTornadovm) {
        return new Qwen35ModelLoader(
                source.gguf().getFileChannel(), source.gguf(), contextLength, useTornadovm);
    }
}
