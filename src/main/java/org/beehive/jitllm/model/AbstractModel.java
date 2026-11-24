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
package org.beehive.jitllm.model;

import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.tokenizer.Tokenizer;

public abstract class AbstractModel implements Model {

    protected final Tokenizer tokenizer;
    protected final Weights weights;
    protected final ChatFormat chatFormat;

    protected AbstractModel(Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) {
        this.tokenizer = tokenizer;
        this.weights = weights;
        this.chatFormat = chatFormat;
    }

    // Common methods across models

    public Weights weights() {
        return weights;
    }

    public ChatFormat chatFormat() {
        return chatFormat;
    }
}
