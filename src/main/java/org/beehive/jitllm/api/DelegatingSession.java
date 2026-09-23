package org.beehive.jllm.api;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jllm.inference.sampler.Sampler;
import org.beehive.jllm.model.Model;
import org.beehive.jllm.model.format.ChatFormat;
import org.beehive.jllm.runtime.diagnostics.DiagnosticCode;
import org.beehive.jllm.runtime.kv.KvLease;
import org.beehive.jllm.runtime.metrics.MetricKey;
import org.beehive.jllm.runtime.metrics.MetricsReport;
import org.beehive.jllm.runtime.policy.ExecutionPolicy;

/**
 * One sequence, on today's {@code State} and execution plan.
 *
 * <p>Package-private: reached only as {@link GenerationSession}.
 *
 * <p>Not thread-safe, as the interface says. The session owns its {@code State} and, on the GPU
 * path, its execution plan — which is why {@link #close()} matters and why the model refuses to
 * close underneath it.
 */
final class DelegatingSession implements GenerationSession {

    private final DelegatingModel owner;
    private final Model model;
    private final boolean gpu;
    private final int contextLength;

    /**
     * How this session gets its state and its plan — legacy (owns both) or lowered (borrows both
     * from a binding domain).
     *
     * <p>A typed seam rather than a nullable {@code State} and a mode flag: what {@code close()}
     * releases differs between the two, and putting that difference in the types is what stops a
     * lowered session freeing device memory its siblings are still executing in.
     */
    private final SessionRuntime runtime;

    /** This sequence's claim on KV storage, from the session runtime the model handle owns. */
    private final KvLease lease;

    /** The only thing in the facade that names a chat format. */
    private final ConversationEncoder encoder;

    private int position;
    private boolean started;
    private boolean closed;

    /**
     * The exact tokens this session has fed to the model and got back from it, in order.
     *
     * <p><b>The unit of prefix reuse</b> [A1]. It is compared against the complete encoded model
     * input of the next {@code messages(.)} request — which includes the messages, the tool
     * specifications, the model-owned formatting and control tokens, the system content and every
     * previous assistant, tool-call and tool-result turn. Comparing messages instead would miss the
     * case that matters most: attaching a tool changes the encoded system content, so the input
     * diverges from its first tokens while the message list looks unchanged.
     *
     * <p>It tracks {@link #position} exactly, because position is what the key/value cache holds.
     * Grown only where position is.
     */
    private final EncodedPrefix encodedContext = new EncodedPrefix();

    /**
     * Tokens the next prompt-form turn must feed before its own: a previous answer whose reasoning
     * was dropped from the cache, and — after a family that cannot rewind was reset — everything
     * before it. Empty when the cache already holds the whole history.
     */
    private List<Integer> carriedHistory = List.of();

    /**
     * How this session executes, resolved once at construction.
     *
     * <p>A value, not a lookup: the choices it carries were {@code static final} fields read from
     * system properties at class initialization, which no session could differ on and no test could
     * set in time. Nothing reads this per token.
     */
    private final ExecutionPolicy executionPolicy;

    /** Resolved once at construction; never re-read per generation. */
    private final ThinkingMode thinkingMode;

    DelegatingSession(
            DelegatingModel owner,
            Model model,
            boolean gpu,
            int contextLength,
            KvLease lease,
            ExecutionPolicy executionPolicy,
            ThinkingMode thinkingMode) {
        this.thinkingMode = thinkingMode;
        this.owner = owner;
        this.model = model;
        this.gpu = gpu;
        this.contextLength = contextLength;
        this.lease = lease;
        this.encoder = new ConversationEncoder(model, thinkingMode);
        this.executionPolicy = executionPolicy;
        // The runtime decides what this session owns: its own state and plan, or a borrowed
        // workspace and a shared program.
        this.runtime = owner.newRuntime(model, lease, executionPolicy);
        // The runtime resolves the policy into whichever state it owns, before any plan is built:
        // a standalone session into its own, a lowered one not at all, because it borrows the
        // domain's workspace and only does so when the policies already agree.
    }

    /** This session's resolved policy. Package-private: it is not facade v1 surface. */
    ExecutionPolicy executionPolicy() {
        return executionPolicy;
    }

    /**
     * The plan this session has been generating with, or {@code null} if it has not built one.
     * Package-private: not facade v1 surface, and never builds one, because asking {@link
     * SessionRuntime#plan()} would construct a plan on a session that never executed.
     */
    org.beehive.jllm.backend.tornado.TornadoVMMasterPlan planIfBuilt() {
        return runtime.hasPlan() ? runtime.plan() : null;
    }

    @Override
    public org.beehive.jllm.runtime.backend.ExecutionInfo prepare() {
        ensureUsable();
        if (gpu) {
            return runtime.plan().executionInfo();
        }
        return new org.beehive.jllm.runtime.backend.ExecutionInfo(
                "CPU",
                System.getProperty("os.arch")
                        + " / "
                        + Runtime.getRuntime().availableProcessors()
                        + " logical processors",
                "Java " + System.getProperty("java.version"),
                "single-token",
                1,
                "FP32",
                "CPU kernels (no tensor-core MMA)",
                "CPU kernels",
                false,
                false);
    }

    @Override
    public GenerationResult generate(GenerationRequest request) {
        ensureUsable();

        boolean promptForm = request.messages() == null;
        List<Integer> promptTokens =
                promptForm ? encodePrompt(request) : encodeConversation(request);
        if (position + promptTokens.size() >= contextLength) {
            return new GenerationResult(
                    "",
                    promptTokens.size(),
                    0,
                    FinishReason.CONTEXT_FULL,
                    new GenerationTimings(Duration.ZERO, Duration.ZERO, promptTokens.size(), 0));
        }

        // Consumed by this turn's prompt, or superseded by a conversation-form request.
        carriedHistory = List.of();

        Sampler sampler =
                Sampler.selectSampler(
                        model.configuration().vocabularySize(),
                        request.temperature(),
                        request.topP(),
                        request.seed() != null ? request.seed() : System.nanoTime());

        // Tool-aware stop handling is a consequence of attaching tools, never an independent
        // choice, and the facade never exposes a stop-token set [§6].
        Set<Integer> stopTokens = encoder.stopTokens(!request.tools().isEmpty());

        TokenEventStream events =
                new TokenEventStream(
                        model.tokenizer(), stopTokens, request.onEvent(), request.onToken());
        IntConsumer onToken = events::accept;

        // The session's logical values become current in whatever state it executes with. For a
        // borrowed workspace that is what keeps two sessions' conversations apart.
        runtime.beginTurn();

        // The loops take a position bound, not a token count: the budget is the positions the
        // prompt occupies plus the tokens asked for, capped by the session's context — exceeding
        // it would write past the key/value cache this session was sized for. A prompt that
        // opens with the token the state is seeded with occupies one position less (it is fed
        // once), so it is counted by what is ingested, not by its length; otherwise that position
        // would become one more generated token than the request asked for.
        int ingested =
                promptTokens.size()
                        - org.beehive.jllm.inference.PromptIngestion.of(
                                        runtime.executionState(), promptTokens, position)
                                .firstIndex();
        int budget = Math.min(position + ingested + request.maxNewTokens(), contextLength);
        List<Integer> responseTokens;
        if (gpu) {
            // The runtime already decided what this session executes with — its own state and plan,
            // or a binding domain's workspace and shared program. No mode check here.
            responseTokens =
                    runtime.generateOnGpu(
                            model, position, promptTokens, stopTokens, budget, sampler, onToken);
        } else {
            responseTokens =
                    model.generateTokens(
                            runtime.executionState(),
                            position,
                            promptTokens,
                            stopTokens,
                            budget,
                            sampler,
                            false,
                            onToken);
        }

        runtime.endTurn();

        boolean hitStopToken =
                !responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast());
        if (hitStopToken) {
            responseTokens.removeLast();
        }

        int responseStart = position + promptTokens.size();
        position += promptTokens.size() + responseTokens.size();
        // Exactly what the key/value cache now holds, so the next conversation can be compared
        // against it. The terminal stop token is not here for the same reason it is not in
        // position: it was removed above.
        encodedContext.append(promptTokens);
        encodedContext.append(responseTokens);
        started = true;
        if (promptForm) {
            dropReasoningFromHistory(responseStart, responseTokens);
        }

        // The stream drops a held terminal stop token and returns everything else, so the text and
        // the events cannot disagree: concatenating the events' non-empty text is this string.
        String completion = events.finish();

        FinishReason reason =
                hitStopToken
                        ? FinishReason.STOP_TOKEN
                        : position >= contextLength
                                ? FinishReason.CONTEXT_FULL
                                : FinishReason.MAX_TOKENS;

        String truncated = applyStopSequences(completion, request.stopSequences());
        if (truncated.length() < completion.length()) {
            reason = FinishReason.STOP_SEQUENCE;
            completion = truncated;
        }

        // Tool calls are extracted only when tools were supplied: a response that happens to
        // contain tool-shaped text was not answering a tool request, and reading it as one would
        // invent a call the caller never offered.
        List<ChatContent.ToolCall> toolCalls =
                request.tools().isEmpty() ? List.of() : encoder.extractToolCalls(completion);
        if (!toolCalls.isEmpty() && hitStopToken) {
            reason = FinishReason.TOOL_CALL;
        }

        return new GenerationResult(
                completion,
                promptTokens.size(),
                responseTokens.size(),
                reason,
                timings(promptTokens.size(), responseTokens.size()),
                toolCalls);
    }

    /**
     * Stop sequences are applied to the produced text rather than used to halt generation: the
     * underlying loop has no cancellation hook in v1. The result is the same text a caller would
     * have got, at the cost of tokens that were generated and then discarded. The engine tier is
     * where cancellation becomes real.
     */
    private static String applyStopSequences(String text, List<String> stopSequences) {
        int cut = -1;
        for (String stop : stopSequences) {
            if (stop == null || stop.isEmpty()) {
                continue;
            }
            int index = text.indexOf(stop);
            if (index >= 0 && (cut < 0 || index < cut)) {
                cut = index;
            }
        }
        return cut < 0 ? text : text.substring(0, cut);
    }

    /**
     * Timings come from the metrics seam rather than from a stopwatch around the call, so the
     * prefill/decode split is the one the engine actually measured. {@code RunMetrics} is still
     * process-global, so a second session generating concurrently would overwrite these — one more
     * thing the engine tier fixes.
     */
    private GenerationTimings timings(int promptTokens, int generatedTokens) {
        MetricsReport report = org.beehive.jllm.auxiliary.RunMetrics.report();
        return new GenerationTimings(
                Duration.ofNanos(report.valueOr(MetricKey.PREFILL_TIME, 0L)),
                Duration.ofNanos(report.valueOr(MetricKey.DECODE_TIME, 0L)),
                promptTokens,
                generatedTokens);
    }

    /**
     * The model's own chat template, applied by the model. The first turn opens the conversation;
     * later turns continue it, which is why the beginning-of-text and the system prompt appear once
     * and not per request.
     */
    private List<Integer> encodePrompt(GenerationRequest request) {
        ChatFormat chatFormat = model.chatFormat();
        List<Integer> tokens = new ArrayList<>();
        tokens.addAll(carriedHistory);
        if (started) {
            // The previous turn's response is in the cache without its terminator: the stop token
            // that ended it was dropped above, and a response cut by the budget never had one. A
            // new user turn written straight after it would continue an assistant message that
            // was never closed, and the model answers that malformed history by ending its next
            // turn early. Close it the way this format closes any assistant message, which is
            // also what the conversation form encodes, so the two forms keep one history.
            tokens.addAll(assistantTurnTerminator(chatFormat));
        } else {
            if (model.shouldAddBeginOfText()) {
                tokens.add(chatFormat.getBeginOfText());
            }
            if (model.shouldAddSystemPrompt() && request.systemPrompt() != null) {
                tokens.addAll(
                        chatFormat.encodeMessage(
                                new ChatFormat.Message(
                                        ChatFormat.Role.SYSTEM, request.systemPrompt())));
            }
        }
        tokens.addAll(
                chatFormat.encodeMessage(
                        new ChatFormat.Message(ChatFormat.Role.USER, request.prompt())));
        tokens.addAll(
                chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        // The prompt form gets the same reasoning control as the conversation form: it is session
        // configuration, and which request shape was used does not change it.
        encoder.appendThinkingControl(tokens);
        return tokens;
    }

    /**
     * Keeps only the answer of a reasoning turn in the history the next prompt-form turn sees.
     *
     * <p>A reasoning template renders earlier assistant turns without their reasoning, and the
     * model is trained on that shape. Left in the cache, the finished {@code <think>} block makes
     * it close its next turn right after reasoning, without an answer. The conversation form never
     * has the problem: it re-encodes the history the caller sends, reasoning and all as plain text.
     *
     * <p>A family whose state is positional drops the block by rewinding to where the response
     * began; the answer is fed again with the next turn. One that also carries recurrent state
     * cannot rewind, because that state has already consumed the reasoning, so it is reset and the
     * whole history, answer included, is fed again.
     */
    private void dropReasoningFromHistory(int responseStart, List<Integer> responseTokens) {
        int reasoningEnd = responseTokens.indexOf(model.chatFormat().reasoningEndToken());
        if (model.chatFormat().reasoningEndToken() < 0 || reasoningEnd < 0) {
            return;
        }
        int answerStart = reasoningEnd + 1;
        // The template strips the whitespace that separates reasoning from the answer.
        while (answerStart < responseTokens.size()
                && model.tokenizer().decode(List.of(responseTokens.get(answerStart))).isBlank()) {
            answerStart++;
        }
        List<Integer> answer =
                List.copyOf(responseTokens.subList(answerStart, responseTokens.size()));
        if (model.configuration().recurrentStateBytes() == 0) {
            position = responseStart;
            encodedContext.truncate(responseStart);
            carriedHistory = answer;
            return;
        }
        List<Integer> history = new ArrayList<>(encodedContext.tokens().subList(0, responseStart));
        history.addAll(answer);
        position = 0;
        encodedContext.clear();
        runtime.reset();
        carriedHistory = history;
    }

    /**
     * The tokens that end an assistant message in this format: an empty assistant message minus its
     * header. Empty when the format does not render a message as its header followed by the
     * content, since no suffix can then be isolated.
     */
    private static List<Integer> assistantTurnTerminator(ChatFormat chatFormat) {
        List<Integer> header =
                chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, ""));
        List<Integer> message =
                chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, ""));
        if (message.size() < header.size() || !message.subList(0, header.size()).equals(header)) {
            return List.of();
        }
        return message.subList(header.size(), message.size());
    }

    /**
     * Encodes the whole conversation, and feeds the model only the part it does not already hold.
     *
     * <p>The session is a <b>cache, not a memory</b> [§2]: what it retains is derived from the
     * request and never added to it. So the encoded input is computed in full, every time, and the
     * only question is how much of it has to be sent again.
     *
     * <p><b>Reuse requires an exact encoded-token prefix.</b> Any divergence — an edited earlier
     * turn, an assistant turn the caller rewrote, a tool specification that changed the system
     * content — resets the sequence and re-encodes from the start. Correctness does not depend on
     * detecting <i>why</i> it diverged, only on refusing to reuse when it did.
     */
    private List<Integer> encodeConversation(GenerationRequest request) {
        List<Integer> full = encoder.encode(request.messages(), request.tools());
        if (encodedContext.isEmpty()) {
            // Nothing retained: a fresh session, or one just reset. There is nothing to diverge
            // from and nothing to undo, so the sequence is left exactly as the model set it up.
            return full;
        }
        List<Integer> suffix = encodedContext.reusableSuffixOf(full);
        if (suffix != null) {
            return suffix;
        }
        resetForDivergence();
        return full;
    }

    /**
     * A transparent reset: the conversation diverged, so the retained prefix is worthless.
     *
     * <p>Not an error, and not visible to the caller except in timing. The same reset a caller can
     * ask for explicitly, minus the {@code ensureUsable()} that would be redundant here.
     */
    private void resetForDivergence() {
        carriedHistory = List.of();
        position = 0;
        started = false;
        encodedContext.clear();
        carriedHistory = List.of();
        runtime.reset();
    }

    @Override
    public int position() {
        return position;
    }

    @Override
    public void reset() {
        ensureUsable();
        // Reset the sequence in place: position to zero, and the last token forgotten.
        //
        // The key/value contents are deliberately not cleared and the state is not replaced.
        // Attention reads the cache from 0 to the current position, so anything above the
        // position is unreachable — starting the sequence again makes the old contents
        // unreadable rather than merely stale.
        //
        // Keeping the state also keeps the execution plan valid, because the plan is bound to
        // this state's buffers. Replacing either meant tearing the plan down and rebuilding it
        // on the next generate, which TornadoVM refuses once the plan has been warmed up:
        //
        //     TornadoFailureException: reset() was called after warmup() on device
        //
        // so reset() left a GPU session unusable. It also made "start the conversation over"
        // cost a recompilation and a fresh device allocation, which is not what it means.
        position = 0;
        started = false;
        encodedContext.clear();
        // Only this session's logical state. The lowered runtime deliberately does not reach into
        // the shared workspace here — that would be another session's turn.
        runtime.reset();
    }

    @Override
    public void close() {
        if (closed) {
            return; // idempotent
        }
        closed = true;
        runtime.close(); // releases what this session owns, never what it borrows
        owner.sessionClosed(this);
    }

    private void ensureUsable() {
        if (closed) {
            throw new IllegalStateException(
                    DiagnosticCode.USED_AFTER_CLOSE.message("session is closed"));
        }
        if (owner.isClosed()) {
            throw new IllegalStateException(
                    DiagnosticCode.USED_AFTER_CLOSE.message(
                            "the model this session belongs to is closed"));
        }
    }
}
