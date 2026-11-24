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
package org.beehive.jitllm.arch.fixture.program;

import org.beehive.jitllm.tokenizer.Tokenizer;

/**
 * Rule 14's fixture: a core-layer type that requires a tokenizer.
 *
 * <p>A program component that cannot be constructed without one has decided that every model
 * generates text, which is the assumption Rule 14 exists to keep out of the core.
 */
public final class ViolatingGenerationAssumption {

    private final Tokenizer tokenizer;

    public ViolatingGenerationAssumption(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }

    public Tokenizer tokenizer() {
        return tokenizer;
    }
}
