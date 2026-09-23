package org.beehive.jllm.server;

import java.util.List;
import org.beehive.jllm.api.ChatMessage;
import org.beehive.jllm.api.FinishReason;
import org.beehive.jllm.api.GenerationRequest;
import org.beehive.jllm.api.GenerationResult;
import org.beehive.jllm.api.GenerationSession;
import org.beehive.jllm.api.LocalModel;
import org.beehive.jllm.api.TextGenerationModel;

/**
 * Reusable, thread-safe inference wrapper over one loaded model.
 *
 * <p>Holds a single {@link GenerationSession} and serializes generation on it — the GPU is a single
 * context, so requests run one at a time. The session owns the KV lease, the execution plan, the
 * chat template and the incremental decode, which is why none of that appears here.
 *
 * <p>Each request is independent: the server's endpoints carry the whole conversation in {@code
 * messages}, so the session is {@link GenerationSession#reset() reset} before every generation
 * rather than accumulating history of its own.
 */
public final class InferenceService {

    private final LocalModel model;
    private final GenerationSession session;
    private final Object lock = new Object();

    public InferenceService(LocalModel model) {
        this.model = model;
        this.session = ((TextGenerationModel) model).newSession();
    }

    public org.beehive.jllm.runtime.backend.ExecutionInfo prepare() {
        synchronized (lock) {
            return session.prepare();
        }
    }

    public LocalModel model() {
        return model;
    }

    /**
     * A single generation request. {@code messages} are role/content turns (chat); a lone user turn
     * covers the completions endpoint.
     */
    public record Request(
            List<ChatMessage> messages, int maxTokens, float temperature, float topP, long seed) {}

    /** Result of a generation: the text and the token counts (for {@code usage}). */
    public record Result(String text, int promptTokens, int completionTokens, boolean stopped) {}

    /**
     * Generate a completion. If {@code onToken} is non-null each decoded token's text is streamed
     * to it as it is produced; the full text is always returned. Thread-safe (serialized).
     */
    public Result generate(Request req, java.util.function.Consumer<String> onToken) {
        synchronized (lock) {
            session.reset();
            GenerationRequest.Builder builder =
                    GenerationRequest.builder()
                            .messages(req.messages())
                            .maxNewTokens(req.maxTokens() > 0 ? req.maxTokens() : 256)
                            .temperature(req.temperature())
                            .topP(req.topP())
                            .seed(req.seed());
            if (onToken != null) {
                builder.onEvent(
                        event -> {
                            if (!event.text().isEmpty()) {
                                onToken.accept(event.text());
                            }
                        });
            }
            GenerationResult result = session.generate(builder.build());
            return new Result(
                    result.text(),
                    result.promptTokens(),
                    result.generatedTokens(),
                    result.finishReason() == FinishReason.STOP_TOKEN);
        }
    }

    /** Close the session; the model outlives it and is closed by the caller that loaded it. */
    public void close() {
        synchronized (lock) {
            session.close();
        }
    }
}
