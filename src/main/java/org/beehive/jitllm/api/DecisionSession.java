package org.beehive.jitllm.api;

import org.beehive.jitllm.inference.PromptIngestion;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.model.Model;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Scores continuations instead of generating them: prefill a prefix once, then read the next-token
 * logits of a few candidate tokens after each of several branches of it.
 *
 * <p>This is the primitive behind "System One" decision endpoints (a state plus typed questions,
 * answered with probabilities rather than text): the state is the prefix, each question is a
 * branch, and the candidates are the tokens that label the question's options. No token is kept;
 * the only output is logits.
 *
 * <h2>How a branch reuses the prefix</h2>
 *
 * <p>A turn feeds the session's <i>seed</i> at its start position, then its prompt, and writes a
 * KV entry for each. The first branch is ingested together with the prefix from position 0. Every
 * later branch starts at the first position after the prefix, with the seed set to its own first
 * token ({@link SessionRuntime#reseed}), so it overwrites the previous branch's KV entries and
 * attends to the prefix the first call wrote. Prefix cost is paid once per {@link #score} call.
 *
 * <p>One session, one caller at a time, like {@link GenerationSession}. Each {@link #score} call
 * starts from an empty context.
 */
@Experimental
public final class DecisionSession implements AutoCloseable {

    private final DelegatingSession session;
    private final SessionRuntime runtime;
    private final Model model;
    private final boolean gpu;
    private final int contextLength;

    private DecisionSession(DelegatingSession session) {
        this.session = session;
        this.runtime = session.runtime();
        this.model = session.model();
        this.gpu = session.onGpu();
        this.contextLength = session.contextLength();
    }

    /**
     * Opens a decision session on {@code model}. It holds one of the model's sessions (and its KV
     * lease) until closed.
     */
    public static DecisionSession open(TextGenerationModel model, SessionOptions options) {
        GenerationSession opened = model.newSession(options);
        if (!(opened instanceof DelegatingSession delegating)) {
            opened.close();
            throw new IllegalArgumentException(
                    "decision scoring needs a model loaded through LocalModels; got "
                            + model.getClass().getName());
        }
        return new DecisionSession(delegating);
    }

    /**
     * Opens a decision session with thinking switched off where the chat format can say so: a
     * decision is read from the first answer token, and a reasoning block there would be scored
     * instead.
     */
    public static DecisionSession open(TextGenerationModel model) {
        try {
            return open(model, SessionOptions.builder().thinkingMode(ThinkingMode.DISABLED).build());
        } catch (IllegalArgumentException | UnsupportedOperationException noThinkingControl) {
            return open(model, SessionOptions.defaults());
        }
    }

    /** The loaded model, for its tokenizer and chat format. */
    public Model model() {
        return model;
    }

    public int contextLength() {
        return contextLength;
    }

    /**
     * Next-token logits of {@code candidates[b]} after {@code prefix + branches[b]}, for every
     * branch.
     *
     * @param prefix token ids shared by every branch, e.g. the rendered state (may be empty only if
     *     every branch is non-empty)
     * @param branches one token list per question; each must be non-empty
     * @param candidates for each branch, the token ids whose logits to return
     * @return {@code result[b][i]} = logit of {@code candidates[b][i]} after branch {@code b}
     */
    public float[][] score(List<Integer> prefix, List<List<Integer>> branches, int[][] candidates) {
        if (branches.isEmpty()) {
            return new float[0][];
        }
        if (candidates.length != branches.size()) {
            throw new IllegalArgumentException(
                    "one candidate list per branch: " + candidates.length + " vs " + branches.size());
        }
        for (List<Integer> branch : branches) {
            if (branch.isEmpty()) {
                throw new IllegalArgumentException("a branch must contain at least one token");
            }
        }

        session.reset();
        runtime.beginTurn();
        try {
            float[][] out = new float[branches.size()][];

            // Branch 0 carries the prefix: one call from position 0.
            List<Integer> first = new ArrayList<>(prefix.size() + branches.get(0).size());
            first.addAll(prefix);
            first.addAll(branches.get(0));
            PromptIngestion ingestion =
                    PromptIngestion.of(runtime.executionState(), first, 0);
            // Where the prefix ends: the seed occupies position 0 unless the prompt repeats it,
            // in which case prompt[0] is fed as the seed and prompt[i] lands at position i.
            int offset = ingestion.firstIndex() == 1 ? 0 : 1;
            int afterPrefix = prefix.size() + offset;
            int longest = afterPrefix;
            for (List<Integer> branch : branches) {
                longest = Math.max(longest, afterPrefix + branch.size());
            }
            if (longest + 1 >= contextLength) {
                throw new IllegalArgumentException(
                        "state plus the longest question is "
                                + longest
                                + " tokens; the session holds "
                                + contextLength);
            }
            out[0] = runOne(0, first, first.size() - ingestion.firstIndex(), candidates[0]);

            // Later branches: rewind to the end of the prefix, seed with the branch's own first
            // token, ingest the rest.
            for (int b = 1; b < branches.size(); b++) {
                List<Integer> branch = branches.get(b);
                runtime.reseed(branch.get(0));
                List<Integer> rest = branch.subList(1, branch.size());
                out[b] = runOne(afterPrefix, rest, rest.size(), candidates[b]);
            }
            return out;
        } finally {
            runtime.endTurn();
            session.reset();
        }
    }

    /**
     * As {@link #score}, with each branch given as a whole chat conversation rendered through the
     * model's own chat template. The shared prefix is the longest common token prefix of all
     * renderings, so it is correct for any template and any tokenizer merge at the boundary between
     * the shared part (e.g. the state) and the per-question part.
     *
     * @return the logits, and how many tokens were shared vs. ingested per branch
     */
    public Scored scoreConversations(List<List<ChatMessage>> conversations, int[][] candidates) {
        List<List<Integer>> encoded = new ArrayList<>(conversations.size());
        for (List<ChatMessage> conversation : conversations) {
            encoded.add(session.encoder().encode(conversation, List.of()));
        }
        int common = encoded.isEmpty() ? 0 : encoded.get(0).size();
        for (List<Integer> tokens : encoded) {
            int limit = Math.min(common, tokens.size());
            int i = 0;
            while (i < limit && tokens.get(i).equals(encoded.get(0).get(i))) {
                i++;
            }
            common = i;
        }
        // Every branch keeps at least one token of its own.
        for (List<Integer> tokens : encoded) {
            common = Math.min(common, tokens.size() - 1);
        }
        List<Integer> prefix = encoded.isEmpty() ? List.of() : encoded.get(0).subList(0, common);
        List<List<Integer>> branches = new ArrayList<>(encoded.size());
        int branchTokens = 0;
        for (List<Integer> tokens : encoded) {
            List<Integer> branch = tokens.subList(common, tokens.size());
            branches.add(branch);
            branchTokens += branch.size();
        }
        float[][] logits = score(prefix, branches, candidates);
        return new Scored(logits, prefix.size(), branchTokens);
    }

    /**
     * Scores plus the token accounting a caller reports as usage.
     *
     * @param logits {@code logits[b][i]} for candidate {@code i} of branch {@code b}
     * @param sharedTokens prefix tokens prefilled once for all branches
     * @param branchTokens tokens ingested across all branches
     */
    public record Scored(float[][] logits, int sharedTokens, int branchTokens) {}

    /** Ingests {@code prompt} from {@code start} and captures the logits of the one next token. */
    private float[] runOne(int start, List<Integer> prompt, int ingested, int[] wanted) {
        float[] captured = new float[wanted.length];
        boolean[] seen = new boolean[1];
        Sampler capture =
                logits -> {
                    if (!seen[0]) {
                        for (int i = 0; i < wanted.length; i++) {
                            captured[i] = logits.get(wanted[i]);
                        }
                        seen[0] = true;
                    }
                    return wanted.length > 0 ? wanted[0] : 0;
                };
        // One generated token: the loop stops at start + ingested + 1.
        int budget = Math.min(start + ingested + 1, contextLength);
        if (gpu) {
            runtime.generateOnGpu(model, start, prompt, Set.of(), budget, capture, t -> {});
        } else {
            model.generateTokens(
                    runtime.executionState(), start, prompt, Set.of(), budget, capture, false, t -> {});
        }
        if (!seen[0]) {
            throw new IllegalStateException(
                    "no logits were produced at position " + (start + ingested)
                            + "; the generation loop sampled nothing (device-side sampling on?)");
        }
        return captured;
    }

    @Override
    public void close() {
        session.close();
    }
}
