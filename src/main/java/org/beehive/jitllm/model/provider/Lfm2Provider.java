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
import org.beehive.jitllm.model.loader.Lfm2ModelLoader;

/**
 * The {@code lfm2moe} architecture: LFM2-8B-A1B and LFM2.5-24B-A2B.
 *
 * <p>Recognition needs no case in {@link GgufRecognition}: the files declare {@code
 * general.architecture = lfm2moe}, and nothing else claims that name.
 */
public final class Lfm2Provider extends FamilyProvider {

    public Lfm2Provider() {
        super("lfm2moe");
    }

    @Override
    protected Lfm2ModelLoader loader(ModelSource source, int contextLength, boolean useTornadovm) {
        return new Lfm2ModelLoader(
                source.gguf().getFileChannel(), source.gguf(), contextLength, useTornadovm);
    }
}
