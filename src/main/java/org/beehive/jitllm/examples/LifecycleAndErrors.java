package org.beehive.jitllm.examples;

import java.nio.file.Path;
import org.beehive.jitllm.api.GenerationRequest;
import org.beehive.jitllm.api.GenerationSession;
import org.beehive.jitllm.api.LocalModel;
import org.beehive.jitllm.api.LocalModels;
import org.beehive.jitllm.api.ModelOptions;
import org.beehive.jitllm.api.TextGenerationModel;

/**
 * What the API does when you get the order wrong, and how to tell why.
 *
 * <p>Every failure carries a {@code DiagnosticCode} in its message, so a caller can branch on the
 * cause without parsing prose. This example provokes the two lifecycle codes on purpose:
 *
 * <ul>
 *   <li>{@code GPUL-LIFE-002} — closing a model while a session is still open. The model stays open
 *       and usable, and no session is force-closed, so the fix is to close the sessions and try
 *       again.
 *   <li>{@code GPUL-LIFE-001} — using a session after it has been closed.
 * </ul>
 */
public final class LifecycleAndErrors {

    private LifecycleAndErrors() {}

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: LifecycleAndErrors <model.gguf>");
            System.exit(2);
        }

        LocalModel model = LocalModels.load(Path.of(args[0]), ModelOptions.defaults());
        GenerationSession session = ((TextGenerationModel) model).newSession();

        try {
            model.close();
            System.out.println("unexpected: the model closed with a session still open");
        } catch (IllegalStateException e) {
            // Names the offending sessions, so a leak is identifiable rather than merely reported.
            System.out.println("refused, as it should be: " + e.getMessage());
        }

        // The failed close had no effect: the model is still usable.
        System.out.println(
                session.generate(
                                GenerationRequest.builder()
                                        .prompt("Still working?")
                                        .maxNewTokens(16)
                                        .build())
                        .text());

        session.close();
        session.close(); // idempotent: a second close is a no-op, not an error

        try {
            session.generate(GenerationRequest.builder().prompt("after close").build());
            System.out.println("unexpected: a closed session generated");
        } catch (IllegalStateException e) {
            System.out.println("closed session refused: " + e.getMessage());
        }

        model.close(); // now that the session is gone, this succeeds
        System.out.println("model closed cleanly");
    }
}
