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
package org.beehive.jitllm.examples;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.api.ChatContent;
import org.beehive.jitllm.api.ChatMessage;
import org.beehive.jitllm.api.ChatRole;
import org.beehive.jitllm.api.FinishReason;
import org.beehive.jitllm.api.GenerationRequest;
import org.beehive.jitllm.api.GenerationResult;
import org.beehive.jitllm.api.GenerationSession;
import org.beehive.jitllm.api.LocalModel;
import org.beehive.jitllm.api.LocalModels;
import org.beehive.jitllm.api.ModelOptions;
import org.beehive.jitllm.api.TextGenerationModel;
import org.beehive.jitllm.api.ToolSpec;

/**
 * A full tool round-trip: describe a tool, let the model call it, run it, hand back the result.
 *
 * <p>The loop is the shape to copy. A model that wants a tool stops with {@link
 * FinishReason#TOOL_CALL} and puts the calls on the result; the caller executes them and appends a
 * {@link ChatContent.ToolResult} for each, then asks again with the conversation so far. Because
 * the tool results have to be interleaved by hand, this is the case that passes {@code messages}
 * explicitly rather than letting the session keep history.
 *
 * <p>Not every family emits tool calls. If this prints an ordinary answer instead, the model simply
 * chose not to call the tool — the API claims tool calling, not forced or named tool choice.
 */
public final class ToolCalling {

    private ToolCalling() {}

    private static final ToolSpec WEATHER =
            new ToolSpec(
                    "get_weather",
                    "Get the current temperature in celsius for a city",
                    """
                    {"type":"object",
                     "properties":{"city":{"type":"string","description":"City name"}},
                     "required":["city"]}
                    """);

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: ToolCalling <model.gguf>");
            System.exit(2);
        }

        List<ChatMessage> conversation = new ArrayList<>();
        conversation.add(ChatMessage.of(ChatRole.USER, "What is the weather in Athens?"));

        try (LocalModel model = LocalModels.load(Path.of(args[0]), ModelOptions.defaults());
                GenerationSession session = ((TextGenerationModel) model).newSession()) {

            for (int round = 0; round < 3; round++) {
                GenerationResult result =
                        session.generate(
                                GenerationRequest.builder()
                                        .messages(conversation)
                                        .tools(List.of(WEATHER))
                                        .maxNewTokens(256)
                                        .temperature(0.0f)
                                        .build());

                if (result.finishReason() != FinishReason.TOOL_CALL) {
                    System.out.println("answer: " + result.text());
                    return;
                }

                // The assistant turn that asked for the tools has to go back in, or the model
                // sees results for calls it never made.
                conversation.add(
                        new ChatMessage(ChatRole.ASSISTANT, List.copyOf(result.toolCalls())));
                for (ChatContent.ToolCall call : result.toolCalls()) {
                    System.out.printf("model called %s(%s)%n", call.name(), call.argumentsJson());
                    String resultJson = execute(call);
                    conversation.add(
                            new ChatMessage(
                                    ChatRole.TOOL,
                                    List.of(
                                            new ChatContent.ToolResult(
                                                    call.id(), call.name(), resultJson))));
                }
            }
            System.out.println("gave up after three tool rounds");
        }
    }

    /** Stands in for real work; a production tool would parse the arguments and call something. */
    private static String execute(ChatContent.ToolCall call) {
        return "{\"temperature_c\": 24, \"conditions\": \"clear\"}";
    }
}
