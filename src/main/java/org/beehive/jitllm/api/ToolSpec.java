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

import java.util.Objects;

/**
 * A tool the model may ask to call.
 *
 * @param name the tool's name, non-blank — what comes back in a {@link ChatContent.ToolCall}
 * @param description what the tool does, for the model to read. May be empty; never {@code null}
 * @param parametersJsonSchema the parameter schema as an <b>opaque JSON string</b>. This milestone
 *     introduces no JSON-schema model: both integrations already hold JSON here, and the engine
 *     only splices it into a prompt
 */
public record ToolSpec(String name, String description, String parametersJsonSchema) {

    public ToolSpec {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("a tool name must not be blank");
        }
        Objects.requireNonNull(description, "description");
        Objects.requireNonNull(parametersJsonSchema, "parametersJsonSchema");
    }
}
