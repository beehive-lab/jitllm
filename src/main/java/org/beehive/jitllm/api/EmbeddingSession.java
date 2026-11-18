package org.beehive.jitllm.api;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.runtime.policy.ExecutionPolicy;
import org.beehive.jitllm.tokenizer.Qwen3Tokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;

/**
 * Text embeddings from a decoder model on the GPU: the vector of a text is the final-normalised
 * hidden state at its last token, L2-normalised (last-token pooling, as Qwen3-Embedding is
 * trained).
 *
 * <p>The session runs the model's ordinary plan with {@link ExecutionPolicy#hiddenStateToHost()}
 * on, so each step also copies the {@code dim}-float hidden state back; nothing is sampled from the
 * logits. The text is ingested from position 0 with the end-of-text token appended, and the
 * capturing sampler stops the loop at its first call, when the hidden state is the last token's.
 *
 * <p>One session, one caller at a time, like {@link GenerationSession}. Each {@link #embed} call
 * starts from an empty context.
 */
@Experimental
public final class EmbeddingSession implements AutoCloseable {

    private final DelegatingSession session;
    private final SessionRuntime runtime;
    private final Model model;
    private final int contextLength;
    private final int dim;
    private final List<Integer> suffix;

    private EmbeddingSession(DelegatingSession session) {
        this.session = session;
        this.runtime = session.runtime();
        this.model = session.model();
        this.contextLength = session.contextLength();
        this.dim = model.configuration().dim();
        this.suffix = endOfText(model.tokenizer());
    }

    /**
     * Opens an embedding session on {@code model}, which must be loaded through {@link LocalModels}
     * with the GPU backend. {@code options}' policy overrides are kept; the hidden-state copy is
     * switched on.
     */
    public static EmbeddingSession open(TextGenerationModel model, SessionOptions options) {
        SessionOptions.Builder withHidden =
                SessionOptions.builder()
                        .executionPolicy(options.executionPolicy().withHiddenStateToHost(true));
        if (options.contextLength() > 0) {
            withHidden.contextLength(options.contextLength());
        }
        GenerationSession opened = model.newSession(withHidden.build());
        if (!(opened instanceof DelegatingSession delegating) || !delegating.onGpu()) {
            opened.close();
            throw new IllegalArgumentException(
                    "embeddings need a model loaded on the GPU through LocalModels; got "
                            + model.getClass().getName());
        }
        return new EmbeddingSession(delegating);
    }

    public static EmbeddingSession open(TextGenerationModel model) {
        return open(model, SessionOptions.defaults());
    }

    /** Width of the vectors this session returns. */
    public int dimension() {
        return dim;
    }

    /** The loaded model, for its tokenizer. */
    public Model model() {
        return model;
    }

    /** Tokens of {@code text} as this session ingests it, before the end-of-text token. */
    public List<Integer> tokens(String text) {
        Tokenizer tokenizer = model.tokenizer();
        if (tokenizer instanceof Qwen3Tokenizer qwen) {
            return qwen.encodeOrdinary(text);
        }
        return tokenizer.encodeAsList(text);
    }

    /**
     * The L2-normalised embedding of {@code text}. Texts longer than the context (less the
     * end-of-text token) are truncated from the end.
     */
    public float[] embed(String text) {
        List<Integer> tokens = new ArrayList<>(tokens(text));
        int room = Math.max(1, contextLength - suffix.size() - 4);
        if (tokens.size() > room) {
            tokens = new ArrayList<>(tokens.subList(0, room));
        }
        tokens.addAll(suffix);
        return embedTokens(tokens);
    }

    /** {@link #embed} of each text, in order. */
    public List<float[]> embedAll(List<String> texts) {
        List<float[]> out = new ArrayList<>(texts.size());
        for (String text : texts) {
            out.add(embed(text));
        }
        return out;
    }

    private float[] embedTokens(List<Integer> tokens) {
        session.reset();
        // The loop takes its first token as the seed and ingests the rest as the prompt.
        runtime.reseed(tokens.get(0));
        List<Integer> prompt = tokens.subList(1, tokens.size());
        float[] vector = new float[dim];
        boolean[] seen = new boolean[1];
        int stop = tokens.get(tokens.size() - 1);
        Sampler capture =
                logits -> {
                    if (!seen[0]) {
                        runtime.executionState().workspace.copyHiddenState(vector);
                        seen[0] = true;
                    }
                    return stop;
                };
        int budget = Math.min(tokens.size() + 3, contextLength);
        runtime.generateOnGpu(model, 0, prompt, Set.of(stop), budget, capture, t -> {});
        if (!seen[0]) {
            throw new IllegalStateException(
                    "no hidden state was produced for " + tokens.size() + " tokens");
        }
        double norm = 0;
        for (float v : vector) {
            norm += (double) v * v;
        }
        float scale = norm > 0 ? (float) (1.0 / Math.sqrt(norm)) : 0f;
        for (int i = 0; i < dim; i++) {
            vector[i] *= scale;
        }
        return vector;
    }

    private static List<Integer> endOfText(Tokenizer tokenizer) {
        Map<String, Integer> special = tokenizer.getSpecialTokens();
        Integer eot = special.get("<|endoftext|>");
        return eot == null ? List.of() : List.of(eot);
    }

    @Override
    public void close() {
        session.close();
    }
}
