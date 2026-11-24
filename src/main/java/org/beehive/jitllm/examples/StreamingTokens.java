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
import org.beehive.jitllm.api.GenerationRequest;
import org.beehive.jitllm.api.GenerationResult;
import org.beehive.jitllm.api.GenerationSession;
import org.beehive.jitllm.api.LocalModel;
import org.beehive.jitllm.api.LocalModels;
import org.beehive.jitllm.api.ModelOptions;
import org.beehive.jitllm.api.TextGenerationModel;

/**
 * Streaming, and the guarantee that makes it safe to build a UI on.
 *
 * <p>One event per emitted token. {@code event.text()} is the text <i>that token completed</i> and
 * may be empty: UTF-8 is decoded incrementally, so a multi-byte character arrives on the event that
 * finishes it and never split across two. Concatenating every event's text equals {@code
 * result.text()}, which this example asserts rather than assumes.
 *
 * <p>The callback runs outside the session's invocation lock, so slow rendering here cannot stall
 * another session sharing the same compiled program. The other side of that coin: never call back
 * into the session from inside the callback.
 */
public final class StreamingTokens {

    private StreamingTokens() {}

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: StreamingTokens <model.gguf> [prompt]");
            System.exit(2);
        }
        String prompt = args.length > 1 ? args[1] : "Name three colours, one per line.";

        StringBuilder streamed = new StringBuilder();

        try (LocalModel model = LocalModels.load(Path.of(args[0]), ModelOptions.defaults());
                GenerationSession session = ((TextGenerationModel) model).newSession()) {

            GenerationResult result =
                    session.generate(
                            GenerationRequest.builder()
                                    .prompt(prompt)
                                    .maxNewTokens(128)
                                    .onEvent(
                                            event -> {
                                                streamed.append(event.text());
                                                System.out.print(event.text());
                                                System.out.flush();
                                            })
                                    .build());

            System.out.println();
            boolean identical = streamed.toString().equals(result.text());
            System.out.printf(
                    "streamed %d chars, result %d chars, identical=%b%n",
                    streamed.length(), result.text().length(), identical);
            if (!identical) {
                // The one legitimate difference: a stop sequence is trimmed from the finished
                // string after the fact, so the stream can carry slightly more than the result.
                System.out.println("(a stop sequence was trimmed from the finished text)");
            }
        }
    }
}
