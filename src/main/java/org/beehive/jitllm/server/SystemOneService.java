package org.beehive.jitllm.server;

import org.beehive.jitllm.api.ChatMessage;
import org.beehive.jitllm.api.ChatRole;
import org.beehive.jitllm.api.DecisionSession;
import org.beehive.jitllm.api.TextGenerationModel;
import org.beehive.jitllm.tokenizer.Tokenizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * {@code POST /v1/systemone}: the "System One" decision contract (a state plus typed questions,
 * answered with probabilities instead of text) served from a local chat model.
 *
 * <p>The wire shape is the one TypeSafe's Jev API and its self-hosted counterparts speak, so an
 * unmodified Jev client pointed at this server works:
 *
 * <pre>
 * {"state": "..." | [...] | {...}, "model": "...", "questions": {
 *    "id": {"type": "noul",   "instructions": "..."},
 *    "id": {"type": "choice", "instructions": "...", "criteria": {"option": "description", ...}},
 *    "id": {"type": "score",  "instructions": "...", "criteria": ["level 0", "level 1", ...]}}}
 * -&gt; {"model": "...", "answers": {
 *    "id": {"type": "noul", "noul": p},
 *    "id": {"type": "choice", "choice": "option", "probabilities": {...}, "confidence": c},
 *    "id": {"type": "score", "score": x, "legend": {...}, "probabilities": {...}, "confidence": c}},
 *    "usage": {"input_tokens": n, "output_tokens": 0}}
 * </pre>
 *
 * <h2>How an answer is computed</h2>
 *
 * <p>Nothing is generated. Each question is rendered through the model's chat template as the
 * state, the question and its options labelled {@code A}, {@code B}, ..., ending where the
 * assistant's answer would start. {@link DecisionSession} prefills the part every question shares
 * once and reads the next-token logits of the option labels after each question; a softmax over
 * those gives the distribution. A yes/no question is a two-option choice ({@code A) Yes},
 * {@code B) No}); a score is the expected level index.
 *
 * <p>These are the relative preferences of a prompted chat model, not calibrated confidences: a
 * model trained for the task (a pointer head, a fitted temperature) is what calibration needs.
 */
final class SystemOneService implements AutoCloseable {

    static final int MAX_OPTIONS = 26;

    private static final String SYSTEM =
            "You are a decision engine. You read a state and answer one question about it by"
                    + " choosing exactly one of the lettered options. Reply with the letter only.";

    private final DecisionSession decisions;
    private final Tokenizer tokenizer;
    private final String modelName;

    SystemOneService(TextGenerationModel model, String modelName) {
        this.decisions = DecisionSession.open(model);
        this.tokenizer = decisions.model().tokenizer();
        this.modelName = modelName;
    }

    /** Thrown for a request the contract rejects; the handler maps it to HTTP 400/422. */
    static final class BadRequest extends RuntimeException {
        final int status;

        BadRequest(int status, String message) {
            super(message);
            this.status = status;
        }
    }

    private record Question(String id, String type, String instructions, List<String> keys,
            List<String> descriptions) {}

    synchronized Map<String, Object> evaluate(Map<String, Object> body) {
        Object stateValue = body.get("state");
        if (stateValue == null) {
            throw new BadRequest(400, "'state' is required");
        }
        String state = renderState(stateValue);
        if (!(body.get("questions") instanceof Map<?, ?> rawQuestions) || rawQuestions.isEmpty()) {
            throw new BadRequest(400, "'questions' must be a non-empty object");
        }

        List<Question> questions = new ArrayList<>();
        for (Map.Entry<?, ?> e : rawQuestions.entrySet()) {
            questions.add(parseQuestion(String.valueOf(e.getKey()), e.getValue()));
        }

        List<List<ChatMessage>> conversations = new ArrayList<>(questions.size());
        int[][] candidates = new int[questions.size()][];
        int[][] optionOfCandidate = new int[questions.size()][];
        for (int q = 0; q < questions.size(); q++) {
            Question question = questions.get(q);
            conversations.add(
                    List.of(
                            ChatMessage.of(ChatRole.SYSTEM, SYSTEM),
                            ChatMessage.of(ChatRole.USER, renderQuestion(state, question))));
            // Each option label, with and without a leading space: which one a template produces
            // after the assistant header depends on the tokenizer, and both are the same answer.
            List<Integer> ids = new ArrayList<>();
            List<Integer> owner = new ArrayList<>();
            for (int o = 0; o < question.keys().size(); o++) {
                Set<Integer> variants = new LinkedHashSet<>();
                for (String spelling : List.of(label(o), " " + label(o))) {
                    List<Integer> tokens = tokenizer.encodeAsList(spelling);
                    if (!tokens.isEmpty()) {
                        variants.add(tokens.get(0));
                    }
                }
                for (int id : variants) {
                    ids.add(id);
                    owner.add(o);
                }
            }
            candidates[q] = ids.stream().mapToInt(Integer::intValue).toArray();
            optionOfCandidate[q] = owner.stream().mapToInt(Integer::intValue).toArray();
        }

        DecisionSession.Scored scored;
        try {
            scored = decisions.scoreConversations(conversations, candidates);
        } catch (IllegalArgumentException tooLong) {
            throw new BadRequest(422, tooLong.getMessage());
        }

        Map<String, Object> answers = new LinkedHashMap<>();
        for (int q = 0; q < questions.size(); q++) {
            Question question = questions.get(q);
            double[] p = optionProbabilities(scored.logits()[q], optionOfCandidate[q],
                    question.keys().size());
            answers.put(question.id(), answer(question, p));
        }

        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("input_tokens", scored.sharedTokens() + scored.branchTokens());
        usage.put("output_tokens", 0);
        usage.put("shared_prefix_tokens", scored.sharedTokens());

        Map<String, Object> out = new LinkedHashMap<>();
        out.put("model", modelName);
        out.put("answers", answers);
        out.put("usage", usage);
        return out;
    }

    // ── request parsing ──────────────────────────────────────────────────────

    private static Question parseQuestion(String id, Object raw) {
        if (!(raw instanceof Map<?, ?> m)) {
            throw new BadRequest(400, "question '" + id + "' must be an object");
        }
        String type = String.valueOf(m.get("type"));
        Object instructionsValue = m.get("instructions");
        if (instructionsValue == null) {
            throw new BadRequest(400, "question '" + id + "' needs 'instructions'");
        }
        String instructions = instructionsValue instanceof String s ? s : Json.write(instructionsValue);
        List<String> keys = new ArrayList<>();
        List<String> descriptions = new ArrayList<>();
        switch (type) {
            case "noul" -> {
                keys.add("yes");
                descriptions.add("Yes");
                keys.add("no");
                descriptions.add("No");
            }
            case "choice" -> {
                Object criteria = m.get("criteria");
                if (criteria instanceof Map<?, ?> options) {
                    for (Map.Entry<?, ?> o : options.entrySet()) {
                        keys.add(String.valueOf(o.getKey()));
                        descriptions.add(o.getKey() + ": " + o.getValue());
                    }
                } else if (criteria instanceof List<?> options) {
                    for (Object o : options) {
                        keys.add(String.valueOf(o));
                        descriptions.add(String.valueOf(o));
                    }
                } else {
                    throw new BadRequest(400,
                            "choice question '" + id + "' needs 'criteria' (object or array)");
                }
                if (keys.size() < 2) {
                    throw new BadRequest(422, "choice question '" + id + "' needs at least 2 options");
                }
            }
            case "score" -> {
                if (!(m.get("criteria") instanceof List<?> levels)) {
                    throw new BadRequest(400, "score question '" + id + "' needs 'criteria' (array)");
                }
                if (levels.size() < 2 || levels.size() > 10) {
                    throw new BadRequest(422,
                            "score question '" + id + "' needs 2-10 levels, got " + levels.size());
                }
                for (int i = 0; i < levels.size(); i++) {
                    keys.add(Integer.toString(i));
                    descriptions.add(String.valueOf(levels.get(i)));
                }
            }
            default -> throw new BadRequest(422,
                    "question '" + id + "' has unknown type '" + type + "' (noul, choice, score)");
        }
        if (keys.size() > MAX_OPTIONS) {
            throw new BadRequest(422, "question '" + id + "' has more than " + MAX_OPTIONS + " options");
        }
        return new Question(id, type, instructions, keys, descriptions);
    }

    /** A string as-is; a list one item per line; an object one {@code name: value} per line. */
    static String renderState(Object state) {
        if (state instanceof String s) {
            return s;
        }
        StringBuilder sb = new StringBuilder();
        if (state instanceof List<?> items) {
            for (Object item : items) {
                sb.append("- ").append(item instanceof String s ? s : Json.write(item)).append('\n');
            }
        } else if (state instanceof Map<?, ?> fields) {
            for (Map.Entry<?, ?> f : fields.entrySet()) {
                Object v = f.getValue();
                sb.append(f.getKey()).append(": ").append(v instanceof String s ? s : Json.write(v))
                        .append('\n');
            }
        } else {
            sb.append(Json.write(state));
        }
        return sb.toString().strip();
    }

    private static String renderQuestion(String state, Question q) {
        StringBuilder sb = new StringBuilder();
        sb.append("State:\n").append(state).append("\n\n");
        sb.append("Question: ").append(q.instructions()).append('\n');
        if (q.type().equals("score")) {
            sb.append("Pick the level that fits best:\n");
        } else {
            sb.append("Options:\n");
        }
        for (int o = 0; o < q.keys().size(); o++) {
            sb.append(label(o)).append(") ").append(q.descriptions().get(o)).append('\n');
        }
        sb.append("Answer with the letter only.");
        return sb.toString();
    }

    private static String label(int option) {
        return String.valueOf((char) ('A' + option));
    }

    // ── answers ──────────────────────────────────────────────────────────────

    /** Log-sum-exp the spelling variants of each option, then softmax over the options. */
    static double[] optionProbabilities(float[] logits, int[] owner, int options) {
        double[] perOption = new double[options];
        java.util.Arrays.fill(perOption, Double.NEGATIVE_INFINITY);
        for (int i = 0; i < logits.length; i++) {
            double a = perOption[owner[i]];
            double b = logits[i];
            double hi = Math.max(a, b);
            perOption[owner[i]] = hi == Double.NEGATIVE_INFINITY ? hi
                    : hi + Math.log(Math.exp(a - hi) + Math.exp(b - hi));
        }
        double max = Double.NEGATIVE_INFINITY;
        for (double v : perOption) {
            max = Math.max(max, v);
        }
        double sum = 0;
        double[] p = new double[options];
        for (int o = 0; o < options; o++) {
            p[o] = Math.exp(perOption[o] - max);
            sum += p[o];
        }
        for (int o = 0; o < options; o++) {
            p[o] /= sum;
        }
        return p;
    }

    private static Map<String, Object> answer(Question q, double[] p) {
        Map<String, Object> a = new LinkedHashMap<>();
        a.put("type", q.type());
        int best = 0;
        for (int o = 1; o < p.length; o++) {
            if (p[o] > p[best]) {
                best = o;
            }
        }
        switch (q.type()) {
            case "noul" -> a.put("noul", round(p[0]));
            case "choice" -> {
                a.put("choice", q.keys().get(best));
                Map<String, Object> probs = new LinkedHashMap<>();
                for (int o = 0; o < p.length; o++) {
                    probs.put(q.keys().get(o), round(p[o]));
                }
                a.put("probabilities", probs);
                a.put("confidence", round(p[best]));
            }
            default -> { // score
                double expected = 0;
                Map<String, Object> legend = new LinkedHashMap<>();
                Map<String, Object> probs = new LinkedHashMap<>();
                for (int o = 0; o < p.length; o++) {
                    expected += o * p[o];
                    legend.put(q.keys().get(o), q.descriptions().get(o));
                    probs.put(q.keys().get(o), round(p[o]));
                }
                a.put("score", round(expected));
                a.put("legend", legend);
                a.put("probabilities", probs);
                a.put("confidence", round(p[best]));
            }
        }
        return a;
    }

    private static double round(double v) {
        return Math.round(v * 1e4) / 1e4;
    }

    @Override
    public void close() {
        decisions.close();
    }
}
