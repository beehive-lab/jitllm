package org.beehive.jllm.examples;

import java.nio.file.Path;
import org.beehive.jllm.api.GenerationRequest;
import org.beehive.jllm.api.GenerationSession;
import org.beehive.jllm.api.LocalModel;
import org.beehive.jllm.api.LocalModels;
import org.beehive.jllm.api.ModelOptions;
import org.beehive.jllm.api.TextGenerationModel;

/**
 * A multi-turn conversation, and the two ways to hold one.
 *
 * <p><b>The session remembers.</b> Send only the new user turn and the session supplies the rest;
 * {@code reset()} rewinds the conversation without releasing the session's storage, so the next
 * turn starts fresh without paying to rebuild the execution plan.
 *
 * <p>The alternative is to pass the whole conversation as {@code messages} on every request, which
 * {@link ToolCalling} does because a tool round-trip has to. Either works; mixing them in one
 * session does not, because the session would prepend history to a list that already contains it.
 */
public final class Conversation {

    private Conversation() {}

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: Conversation <model.gguf>");
            System.exit(2);
        }

        try (LocalModel model = LocalModels.load(Path.of(args[0]), ModelOptions.defaults());
                GenerationSession session = ((TextGenerationModel) model).newSession()) {

            ask(session, "My favourite number is 7. Remember it.", "Answer in one short sentence.");
            // No system prompt and no restatement: the session still has the first turn.
            ask(session, "What is my favourite number times 6?", null);

            System.out.printf("%n-- reset: the session forgets, the model stays loaded --%n");
            session.reset();
            ask(session, "What is my favourite number?", null);
        }
    }

    private static void ask(GenerationSession session, String prompt, String systemPrompt) {
        GenerationRequest.Builder builder =
                GenerationRequest.builder().prompt(prompt).maxNewTokens(96).temperature(0.0f);
        if (systemPrompt != null) {
            builder.systemPrompt(systemPrompt);
        }
        System.out.printf("%n> %s%n", prompt);
        System.out.println(session.generate(builder.build()).text());
    }
}
