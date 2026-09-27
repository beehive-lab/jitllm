package org.beehive.jitllm.api;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.beehive.jitllm.inference.PromptIngestion;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.model.Model;

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
 * <p>The first branch is ingested together with the prefix from position 0. Every later branch
 * starts at the first position after the prefix, so it overwrites the previous branch's KV entries
 * and attends to the prefix the first call wrote: prefix cost is paid once per {@link #score} call.
 * Where exactly a branch must start, and whether its first token goes in as the session's seed
 * ({@link SessionRuntime#reseed}) or as the first prompt token, differs between generation loops;
 * the session measures it once on a small probe ({@link #calibrate}) and falls back to scoring each
 * branch from position 0 if no convention reproduces the unshared result.
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
            return open(
                    model, SessionOptions.builder().thinkingMode(ThinkingMode.DISABLED).build());
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
                    "one candidate list per branch: "
                            + candidates.length
                            + " vs "
                            + branches.size());
        }
        for (List<Integer> branch : branches) {
            if (branch.isEmpty()) {
                throw new IllegalArgumentException("a branch must contain at least one token");
            }
        }
        int longest = prefix.size() + 1;
        for (List<Integer> branch : branches) {
            longest = Math.max(longest, prefix.size() + 1 + branch.size());
        }
        if (longest + 3 >= contextLength) {
            throw new IllegalArgumentException(
                    "state plus the longest question is "
                            + longest
                            + " tokens; the session holds "
                            + contextLength);
        }
        return scoreWith(convention(), prefix, branches, candidates);
    }

    /**
     * How this model's generation loop places tokens, which is what rewinding to the end of a
     * prefix depends on.
     *
     * <p>Two conventions exist in the engine: most loops feed the session's seed at the start
     * position before the prompt; the Qwen 3 loops ingest the prompt from its first token and use
     * the seed only for decoding. Which one a session gets depends on the family and on whether it
     * runs the lowered or the legacy path, so it is measured rather than assumed (see {@link
     * #calibrate}).
     *
     * @param shared whether branches reuse the prefix at all; {@code false} scores every branch
     *     from position 0, correct for any loop and slower
     * @param seedFed the loop feeds the seed at the start position (so a branch is seeded with its
     *     first token and the rest is the prompt); otherwise the whole branch is the prompt
     * @param offset positions before the first prefix token (1 if the seed occupies position 0)
     */
    private record Convention(boolean shared, boolean seedFed, int offset) {
        static final Convention UNSHARED = new Convention(false, false, 0);
    }

    private Convention convention;

    private Convention convention() {
        if (convention == null) {
            convention = calibrate();
        }
        return convention;
    }

    /**
     * Picks the convention under which a branched score equals the same continuation scored alone
     * from position 0, on a small probe rendered through the model's own chat template. Falls back
     * to {@link Convention#UNSHARED} if none matches.
     */
    private Convention calibrate() {
        List<List<Integer>> probe = new ArrayList<>();
        probe.add(
                session.encoder()
                        .encode(
                                List.of(
                                        ChatMessage.of(
                                                ChatRole.USER,
                                                "Calibration. Reply with one letter. First question: is the sky blue? A) yes B) no")),
                                List.of()));
        probe.add(
                session.encoder()
                        .encode(
                                List.of(
                                        ChatMessage.of(
                                                ChatRole.USER,
                                                "Calibration. Reply with one letter. Second question: is ice hot? A) yes B) no")),
                                List.of()));
        int common = commonPrefix(probe);
        List<Integer> prefix = probe.get(0).subList(0, common);
        List<List<Integer>> branches =
                List.of(
                        probe.get(0).subList(common, probe.get(0).size()),
                        probe.get(1).subList(common, probe.get(1).size()));
        int[] ids = new int[Math.min(16, model.configuration().vocabularySize())];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = branches.get(1).get(i % branches.get(1).size());
        }
        int[][] candidates = {ids, ids};

        float[] reference = scoreWith(Convention.UNSHARED, prefix, branches, candidates)[1];
        Convention best = Convention.UNSHARED;
        double bestError = Double.MAX_VALUE;
        for (boolean seedFed : new boolean[] {true, false}) {
            for (int offset : new int[] {0, 1}) {
                Convention c = new Convention(true, seedFed, offset);
                float[] got = scoreWith(c, prefix, branches, candidates)[1];
                double error = 0;
                for (int i = 0; i < got.length; i++) {
                    error = Math.max(error, Math.abs(got[i] - reference[i]));
                }
                if (error < bestError) {
                    bestError = error;
                    best = c;
                }
            }
        }
        // FP16 accumulation on the device moves logits by a few hundredths; a wrong position moves
        // them by whole units.
        return bestError < 0.25 ? best : Convention.UNSHARED;
    }

    private float[][] scoreWith(
            Convention c, List<Integer> prefix, List<List<Integer>> branches, int[][] candidates) {
        float[][] out = new float[branches.size()][];
        if (!c.shared()) {
            for (int b = 0; b < branches.size(); b++) {
                List<Integer> whole = new ArrayList<>(prefix);
                whole.addAll(branches.get(b));
                out[b] = fromStart(whole, candidates[b]);
            }
            return out;
        }
        session.reset();
        runtime.beginTurn();
        try {
            List<Integer> first = new ArrayList<>(prefix);
            first.addAll(branches.get(0));
            PromptIngestion ingestion = PromptIngestion.of(runtime.executionState(), first, 0);
            out[0] = runOne(0, first, first.size() - ingestion.firstIndex(), candidates[0]);
            int afterPrefix = prefix.size() + c.offset();
            for (int b = 1; b < branches.size(); b++) {
                List<Integer> branch = branches.get(b);
                runtime.reseed(branch.get(0));
                List<Integer> prompt = c.seedFed() ? branch.subList(1, branch.size()) : branch;
                out[b] = runOne(afterPrefix, prompt, prompt.size(), candidates[b]);
            }
            return out;
        } finally {
            runtime.endTurn();
            session.reset();
        }
    }

    /** Scores {@code tokens} as a whole, from an empty context. */
    private float[] fromStart(List<Integer> tokens, int[] candidates) {
        session.reset();
        runtime.beginTurn();
        try {
            PromptIngestion ingestion = PromptIngestion.of(runtime.executionState(), tokens, 0);
            return runOne(0, tokens, tokens.size() - ingestion.firstIndex(), candidates);
        } finally {
            runtime.endTurn();
            session.reset();
        }
    }

    private static int commonPrefix(List<List<Integer>> encoded) {
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
        return Math.max(0, common);
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
        int common = commonPrefix(encoded);
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

    /**
     * Ingests {@code prompt} from {@code start} and captures the logits of the one next token.
     *
     * <p>Where the first sample falls depends on the family: a seed that the prompt does not repeat
     * (Qwen's start header, or a {@link SessionRuntime#reseed reseeded} branch) occupies a position
     * of its own. Rather than mirror each loop's position arithmetic, the budget leaves a little
     * slack and the capturing sampler answers with a token that is the only stop token, so every
     * loop ends right after its first sample.
     */
    private float[] runOne(int start, List<Integer> prompt, int ingested, int[] wanted) {
        float[] captured = new float[wanted.length];
        boolean[] seen = new boolean[1];
        int stop = wanted.length > 0 ? wanted[0] : 0;
        Sampler capture =
                logits -> {
                    if (!seen[0]) {
                        for (int i = 0; i < wanted.length; i++) {
                            captured[i] = logits.get(wanted[i]);
                        }
                        seen[0] = true;
                    }
                    return stop;
                };
        int budget = Math.min(start + ingested + 3, contextLength);
        Set<Integer> stopTokens = Set.of(stop);
        if (gpu) {
            runtime.generateOnGpu(model, start, prompt, stopTokens, budget, capture, t -> {});
        } else {
            model.generateTokens(
                    runtime.executionState(),
                    start,
                    prompt,
                    stopTokens,
                    budget,
                    capture,
                    false,
                    t -> {});
        }
        if (!seen[0]) {
            throw new IllegalStateException(
                    "no logits were produced after position "
                            + (start + ingested)
                            + "; the generation loop sampled nothing (device-side sampling on?)");
        }
        return captured;
    }

    @Override
    public void close() {
        session.close();
    }
}
