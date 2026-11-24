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
package org.beehive.jitllm.model.format;

import java.util.Optional;

/**
 * Represents a single tool call extracted from a model response. Contains the raw strings — JSON
 * parsing of arguments is left to the caller.
 *
 * @param name the tool/function name to invoke
 * @param argumentsJson the arguments as a JSON object string, e.g. {"location":"Boston"}
 * @param id optional tool call ID parsed from the model response; callers that generate IDs
 *     themselves (e.g. Ollama-style "call_XXXXXXXX") may pass {@link Optional#empty()} and let the
 *     consumer generate one
 */
public record ToolCallExtract(String name, String argumentsJson, Optional<String> id) {

    public ToolCallExtract(String name, String argumentsJson) {
        this(name, argumentsJson, Optional.empty());
    }
}
