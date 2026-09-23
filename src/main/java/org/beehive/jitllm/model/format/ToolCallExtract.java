package org.beehive.jllm.model.format;

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
