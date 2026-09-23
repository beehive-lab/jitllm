package org.beehive.jitllm.model.format;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.beehive.jitllm.tokenizer.MistralTokenizer;

public class MistralChatFormat implements ChatFormat {

    protected final MistralTokenizer tokenizer;
    protected final int unknownToken;
    protected final int beginOfText;
    protected final int endOfText;
    protected final int beginOfInstruction;
    protected final int endOfInstruction;
    protected final int toolCalls;
    protected final int beginOfAvailableTools;
    protected final int endOfAvailableTools;
    protected final int beginOfToolResults;
    protected final int endOfToolResults;
    protected final int prefix;
    protected final int middle;
    protected final int suffix;

    public MistralChatFormat(MistralTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
        this.unknownToken = specialTokens.get("<unk>");
        this.beginOfText = specialTokens.get("<s>");
        this.endOfText = specialTokens.get("</s>");
        this.beginOfInstruction = specialTokens.get("[INST]");
        this.endOfInstruction = specialTokens.get("[/INST]");
        this.toolCalls = specialTokens.get("[TOOL_CALLS]");
        this.beginOfAvailableTools = specialTokens.get("[AVAILABLE_TOOLS]");
        this.endOfAvailableTools = specialTokens.get("[/AVAILABLE_TOOLS]");
        this.beginOfToolResults = specialTokens.get("[TOOL_RESULTS]");
        this.endOfToolResults = specialTokens.get("[/TOOL_RESULTS]");
        // Only Codestral supports FIM tokens.
        this.prefix = specialTokens.getOrDefault("[PREFIX]", unknownToken);
        this.suffix = specialTokens.getOrDefault("[SUFFIX]", unknownToken);
        this.middle = specialTokens.getOrDefault("[MIDDLE]", unknownToken);
    }

    @Override
    public int getBeginOfText() {
        return beginOfText;
    }

    @Override
    public Set<Integer> getStopTokens() {
        return Set.of(endOfText);
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(beginOfInstruction);
        tokens.addAll(tokenizer.encodeAsList(message.role().name()));
        tokens.add(endOfInstruction);
        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = encodeHeader(message);
        tokens.addAll(tokenizer.encodeAsList(message.content().strip()));
        tokens.add(endOfInstruction);
        return tokens;
    }

    public List<Integer> encodeFillInTheMiddle(String prefix, String suffix) {
        List<Integer> tokens = new ArrayList<>();
        // dummy - empty string set to comply with encode method signature.
        final Set<String> EMPTY_STRING_SET = Collections.emptySet();
        tokens.add(this.suffix);
        tokens.addAll(tokenizer.encode(suffix, EMPTY_STRING_SET));
        tokens.add(this.prefix);
        tokens.addAll(tokenizer.encode(prefix, EMPTY_STRING_SET));
        return tokens;
    }
}
