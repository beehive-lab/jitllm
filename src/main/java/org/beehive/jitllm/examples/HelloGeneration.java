package org.beehive.jllm.examples;

import java.nio.file.Path;
import org.beehive.jllm.api.GenerationRequest;
import org.beehive.jllm.api.GenerationResult;
import org.beehive.jllm.api.GenerationSession;
import org.beehive.jllm.api.LocalModel;
import org.beehive.jllm.api.LocalModels;
import org.beehive.jllm.api.ModelOptions;
import org.beehive.jllm.api.TextGenerationModel;

/**
 * The smallest complete program: load a model, ask it one question, print the answer.
 *
 * <p>The nesting is the point. A session is closed before its model, and writing it as nested
 * try-with-resources makes that the natural spelling rather than something to remember.
 */
public final class HelloGeneration {

    private HelloGeneration() {}

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: HelloGeneration <model.gguf> [prompt]");
            System.exit(2);
        }
        String prompt = args.length > 1 ? args[1] : "What is the capital of France?";

        try (LocalModel model = LocalModels.load(Path.of(args[0]), ModelOptions.defaults())) {
            // Generation is a capability, not something every model has: newSession() is declared
            // on TextGenerationModel, so a model that only produces embeddings never offers one.
            TextGenerationModel text = (TextGenerationModel) model;

            try (GenerationSession session = text.newSession()) {
                GenerationResult result =
                        session.generate(
                                GenerationRequest.builder()
                                        .prompt(prompt)
                                        .maxNewTokens(128)
                                        .build());

                System.out.println(result.text());
                System.out.printf(
                        "%n[%d prompt tokens, %d generated, finished on %s, %.1f tok/s decode]%n",
                        result.promptTokens(),
                        result.generatedTokens(),
                        result.finishReason(),
                        result.timings().generatedTokensPerSecond());
            }
        }
    }
}
