package org.beehive.jllm.api;

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
