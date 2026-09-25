package org.beehive.jitllm.model.format;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.beehive.jitllm.model.format.Gemma4ToolCalls.Piece;
import org.beehive.jitllm.tokenizer.Gemma4Tokenizer;

/**
 * Chat format for Gemma 4 models.
 *
 * <p>Gemma 4 uses a {@code <|turn>{role}\n. <turn|>} turn structure (the assistant role is spelled
 * "model" in the template), starts conversations with {@code <bos>}, and stops generation on {@code
 * <turn|>} (the model's configured EOS token).
 *
 * <h2>Tool calling</h2>
 *
 * <p>Follows the chat template embedded in the Gemma 4 GGUF files; the wire syntax is in {@link
 * Gemma4ToolCalls}. In outline:
 *
 * <ul>
 *   <li>The tool declarations go at the end of the system turn, after the caller's system text and
 *       with nothing between them; a conversation with no system message gets a system turn holding
 *       only the declarations.
 *   <li>An assistant turn with calls is {@code <|turn>model\n} followed by one {@code
 *       <|tool_call>…<tool_call|>} per call, and the tool results are written <em>into that same
 *       turn</em> as {@code <|tool_response>…<tool_response|>} blocks. The turn stays open: the
 *       model's answer continues it, so no new {@code <|turn>model\n} header is added after the
 *       results, and a following user turn is preceded by {@code <turn|>\n}.
 *   <li>The model ends a call by emitting {@code <|tool_response>} — it hands over to the tool
 *       there — or, after several calls, {@code <eos>}. Both are extra stop tokens when tools are
 *       attached, as in llama.cpp's end-of-generation set for this family ({@code <eos>}, {@code
 *       <turn|>}, {@code <|tool_response>}).
 * </ul>
 *
 * <p>The markers ({@code <|tool>}, {@code <tool|>}, {@code <|tool_call>}, {@code <tool_call|>},
 * {@code <|tool_response>}, {@code <tool_response|>}, {@code <|"|>}) are single tokens in the
 * vocabulary and are encoded as those ids. Text a caller supplies is always encoded as text, so a
 * description or tool result that happens to spell a marker cannot become one.
 */
public class Gemma4ChatFormat implements ChatFormat {

    protected final Gemma4Tokenizer tokenizer;
    protected final int beginOfText;
    protected final int startTurn;
    protected final int endTurn;
    protected final int toolResponseStart;

    /**
     * {@code <eos>}, which the GGUF types as an ordinary token and so is looked up by spelling; the
     * model emits it after a run of calls.
     */
    protected final int endOfSequence;

    /** The marker spellings the tool syntax uses, to their token ids; absent ones are left out. */
    private final Map<String, Integer> markerTokens;

    public Gemma4ChatFormat(Gemma4Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
        this.beginOfText = specialTokens.getOrDefault("<bos>", -1);
        this.startTurn = specialTokens.getOrDefault("<|turn>", -1);
        this.endTurn = specialTokens.getOrDefault("<turn|>", -1);
        this.toolResponseStart = specialTokens.getOrDefault(Gemma4ToolCalls.RESPONSE_OPEN, -1);
        this.endOfSequence = tokenizer.tokenIndex("<eos>");
        Map<String, Integer> markers = new HashMap<>();
        for (String marker : Gemma4ToolCalls.MARKERS) {
            Integer id = specialTokens.get(marker);
            if (id != null) {
                markers.put(marker, id);
            }
        }
        this.markerTokens = Map.copyOf(markers);
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startTurn);
        // The chat template spells the assistant role "model".
        String role = Role.ASSISTANT.equals(message.role()) ? "model" : message.role().name();
        tokens.addAll(tokenizer.encodeAsList(role));
        tokens.addAll(tokenizer.encodeAsList("\n"));
        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = encodeHeader(message);
        tokens.addAll(tokenizer.encodeAsList(message.content().strip()));
        tokens.addAll(endOfTurn());
        return tokens;
    }

    @Override
    public int getBeginOfText() {
        return beginOfText;
    }

    @Override
    public Set<Integer> getStopTokens() {
        return Set.of(endTurn);
    }

    // ── Tool calling ──────────────────────────────────────────────────────────

    /**
     * Only when the vocabulary has every marker the syntax needs: without them the model was not
     * trained on this format, and spelling the markers out as text would be a guess.
     */
    @Override
    public boolean supportsToolCalling() {
        return markerTokens.size() == Gemma4ToolCalls.MARKERS.size() && startTurn != -1;
    }

    /**
     * The declarations as text, markers spelled out; the tokens come from {@link
     * #encodeToolSystemMessage}.
     */
    @Override
    public String toolSystemPromptSuffix(String toolsJson) {
        return Gemma4ToolCalls.asText(Gemma4ToolCalls.renderDeclarations(toolsJson));
    }

    /**
     * {@code <|turn>system\n} + the system text, trimmed + the declarations + {@code <turn|>\n}.
     */
    @Override
    public List<Integer> encodeToolSystemMessage(String systemContent, String toolsJson) {
        List<Integer> tokens = encodeHeader(new Message(Role.SYSTEM, ""));
        if (systemContent != null) {
            tokens.addAll(tokenizer.encodeAsList(systemContent.strip()));
        }
        tokens.addAll(encodePieces(Gemma4ToolCalls.renderDeclarations(toolsJson)));
        tokens.addAll(endOfTurn());
        return tokens;
    }

    /** {@code <|turn>model\n<|tool_call>…<tool_call|>}, left open for the results. */
    @Override
    public List<Integer> encodeToolCallAssistantTurn(ToolCallExtract toolCall) {
        return encodeToolCallAssistantTurn(List.of(toolCall));
    }

    /** One {@code <|turn>model\n} header, then every call; the turn is left open. */
    @Override
    public List<Integer> encodeToolCallAssistantTurn(List<ToolCallExtract> toolCalls) {
        if (toolCalls.isEmpty()) {
            return List.of();
        }
        List<Integer> tokens = encodeHeader(new Message(Role.ASSISTANT, ""));
        tokens.addAll(encodeToolCallContinuation(toolCalls));
        return tokens;
    }

    @Override
    public List<Integer> encodeToolCallContinuation(List<ToolCallExtract> toolCalls) {
        List<Integer> tokens = new ArrayList<>();
        for (ToolCallExtract call : toolCalls) {
            tokens.addAll(encodePieces(Gemma4ToolCalls.renderCall(call)));
        }
        return tokens;
    }

    /**
     * {@code <|tool_response>response:NAME{value:<|"|>RESULT<|"|>}<tool_response|>}, inside the
     * assistant turn that made the call.
     */
    @Override
    public List<Integer> encodeToolResultTurn(String toolCallId, String toolName, String result) {
        return encodePieces(Gemma4ToolCalls.renderResponse(toolName, result));
    }

    @Override
    public boolean toolResultsStayInAssistantTurn() {
        return true;
    }

    @Override
    public List<Integer> encodeAssistantContinuation(String content) {
        List<Integer> tokens = new ArrayList<>(tokenizer.encodeAsList(content.strip()));
        tokens.addAll(endOfTurn());
        return tokens;
    }

    @Override
    public List<Integer> encodeOpenAssistantTurnEnd() {
        return endOfTurn();
    }

    @Override
    public Optional<ToolCallExtract> extractToolCall(String responseText) {
        return Gemma4ToolCalls.parseFirst(responseText);
    }

    @Override
    public List<ToolCallExtract> extractAllToolCalls(String responseText) {
        return Gemma4ToolCalls.parseAll(responseText);
    }

    /**
     * {@code <turn|>}, plus {@code <|tool_response>} and {@code <eos>}, which are where the model
     * ends a call.
     */
    @Override
    public Set<Integer> getToolAwareStopTokens() {
        Set<Integer> stops = new HashSet<>(getStopTokens());
        if (toolResponseStart != -1) {
            stops.add(toolResponseStart);
        }
        if (endOfSequence != -1) {
            stops.add(endOfSequence);
        }
        return Set.copyOf(stops);
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private List<Integer> endOfTurn() {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(endTurn);
        tokens.addAll(tokenizer.encodeAsList("\n"));
        return tokens;
    }

    private List<Integer> encodePieces(List<Piece> pieces) {
        List<Integer> tokens = new ArrayList<>();
        for (Piece piece : pieces) {
            if (piece.special()) {
                Integer id = markerTokens.get(piece.text());
                if (id == null) {
                    throw new IllegalStateException(
                            "the vocabulary has no " + piece.text() + " token");
                }
                tokens.add(id);
            } else {
                tokens.addAll(tokenizer.encodeAsList(piece.text()));
            }
        }
        return tokens;
    }
}
