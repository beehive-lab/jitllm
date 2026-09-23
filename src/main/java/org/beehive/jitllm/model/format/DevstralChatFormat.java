package org.beehive.jllm.model.format;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.beehive.jllm.tokenizer.DevstralTokenizer;

public class DevstralChatFormat implements ChatFormat {

    protected final DevstralTokenizer tokenizer;
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

    public DevstralChatFormat(DevstralTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
        this.unknownToken = specialTokens.getOrDefault("<unk>", -1);
        this.beginOfText = specialTokens.get("<s>");
        this.endOfText = specialTokens.get("</s>");
        this.beginOfInstruction = specialTokens.get("[INST]");
        this.endOfInstruction = specialTokens.get("[/INST]");
        this.toolCalls = specialTokens.getOrDefault("[TOOL_CALLS]", unknownToken);
        this.beginOfAvailableTools = specialTokens.getOrDefault("[AVAILABLE_TOOLS]", unknownToken);
        this.endOfAvailableTools = specialTokens.getOrDefault("[/AVAILABLE_TOOLS]", unknownToken);
        this.beginOfToolResults = specialTokens.getOrDefault("[TOOL_RESULTS]", unknownToken);
        this.endOfToolResults = specialTokens.getOrDefault("[/TOOL_RESULTS]", unknownToken);
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
        final Set<String> EMPTY_STRING_SET = Collections.emptySet();
        tokens.add(this.suffix);
        tokens.addAll(tokenizer.encode(suffix, EMPTY_STRING_SET));
        tokens.add(this.prefix);
        tokens.addAll(tokenizer.encode(prefix, EMPTY_STRING_SET));
        return tokens;
    }
}
