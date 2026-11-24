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

/**
 * The model families jitllm knows, as an internal identifier. Loading goes through the {@code
 * ModelProvider} that claims a file; this enum selects nothing.
 */
public enum ModelType {
    LLAMA_3,
    MISTRAL,
    DEVSTRAL_2,
    QWEN_2,
    QWEN_3,
    QWEN_2_MOE,
    DEEPSEEK_R1_DISTILL_QWEN,
    PHI_3,
    GRANITE,
    QWEN_3_5,
    GEMMA_4,
    DEEPSEEK_2,
    UNKNOWN;

    public boolean isDeepSeekR1() {
        return this == DEEPSEEK_R1_DISTILL_QWEN;
    }
}
