package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.format.Gemma4ChatFormat;
import org.beehive.jitllm.model.format.Gemma4TestVocabulary;
import org.beehive.jitllm.model.format.GraniteChatFormat;
import org.beehive.jitllm.model.format.LlamaChatFormat;
import org.beehive.jitllm.model.format.Qwen3ChatFormat;
import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tokenizer.Qwen3Tokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;
import org.beehive.jitllm.tokenizer.Vocabulary;
import org.junit.Test;

/**
 * What {@link ModelInfo#capabilities()} reports for each family, and that it is the same answer the
 * request path enforces.
 */
public class ModelCapabilitiesTest {

    @Test
    public void gemma4CallsToolsAndHasNoThinkingSwitch() {
        assertEquals(
                new ModelCapabilities(true, false),
                DelegatingModel.capabilitiesOf(Gemma4TestVocabulary.chatFormat()));
    }

    @Test
    public void aGemma4VocabularyWithoutTheToolMarkersReportsNoTools() {
        assertEquals(
                ModelCapabilities.NONE,
                DelegatingModel.capabilitiesOf(
                        new Gemma4ChatFormat(Gemma4TestVocabulary.tokenizer(false))));
    }

    @Test
    public void qwen3CallsToolsAndControlsThinking() {
        assertEquals(new ModelCapabilities(true, true), DelegatingModel.capabilitiesOf(qwen3()));
    }

    @Test
    public void llamaCallsToolsAndHasNoThinkingSwitch() {
        ChatFormat llama =
                new LlamaChatFormat(
                        new SpecialTokensOnly(
                                Map.of(
                                        "<|begin_of_text|>", 0,
                                        "<|start_header_id|>", 1,
                                        "<|end_header_id|>", 2,
                                        "<|eot_id|>", 3,
                                        "<|end_of_text|>", 4,
                                        "<|eom_id|>", 5,
                                        "<|python_tag|>", 6)));
        assertEquals(new ModelCapabilities(true, false), DelegatingModel.capabilitiesOf(llama));
    }

    @Test
    public void aFamilyWithoutToolCallingReportsNone() {
        ChatFormat granite =
                new GraniteChatFormat(
                        new SpecialTokensOnly(
                                Map.of(
                                        "<|start_of_role|>", 0,
                                        "<|end_of_role|>", 1,
                                        "<|end_of_text|>", 2)));
        assertEquals(ModelCapabilities.NONE, DelegatingModel.capabilitiesOf(granite));
    }

    /** The reported flag and the request path's check are the same predicate. */
    @Test
    public void whatIsReportedIsWhatTheEncoderEnforces() {
        ChatFormat granite =
                new GraniteChatFormat(
                        new SpecialTokensOnly(
                                Map.of(
                                        "<|start_of_role|>", 0,
                                        "<|end_of_role|>", 1,
                                        "<|end_of_text|>", 2)));
        ConversationEncoder encoder =
                new ConversationEncoder(
                        new Gemma4ToolConversationTest.FormatOnlyModel(granite),
                        ThinkingMode.DEFAULT);
        assertFalse(DelegatingModel.capabilitiesOf(granite).toolCalling());
        try {
            encoder.encode(
                    List.of(ChatMessage.of(ChatRole.USER, "hi")),
                    List.of(new ToolSpec("t", "", "{\"type\":\"object\"}")));
            throw new AssertionError("a format without tool calling must refuse tools");
        } catch (UnsupportedOperationException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("GPUL-REQ-003"));
        }
    }

    @Test
    public void aModelInfoBuiltWithoutCapabilitiesReportsNone() {
        ModelInfo info =
                new ModelInfo(
                        "m", "llama", 512, Path.of("m.gguf"), Set.of(DataType.Q8_0), DataType.Q8_0);
        assertEquals(ModelCapabilities.NONE, info.capabilities());
        ModelInfo withTools =
                new ModelInfo(
                        "m",
                        "gemma_4",
                        512,
                        Path.of("m.gguf"),
                        Set.of(DataType.Q8_0),
                        DataType.Q8_0,
                        new ModelCapabilities(true, false));
        assertTrue(withTools.capabilities().toolCalling());
        assertFalse(withTools.capabilities().thinkingControl());
    }

    private static ChatFormat qwen3() {
        String[] tokens = {
            "a", "b", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"
        };
        Map<String, Object> metadata =
                Map.of(
                        "tokenizer.ggml.token_type", new int[] {1, 1, 3, 3, 3, 3, 3},
                        "tokenizer.ggml.merges", new String[0]);
        Qwen3Tokenizer tokenizer =
                new Qwen3Tokenizer(metadata, new Vocabulary(tokens, null), false);
        return new Qwen3ChatFormat(
                tokenizer,
                new ChatFormat.ChatTokens(
                        "<|im_start|>", "<|im_end|>", "", "<|endoftext|>", "<|fim_pad|>"));
    }

    /** A tokenizer that is only a special-token table, which is all these constructors read. */
    private record SpecialTokensOnly(Map<String, Integer> specialTokens) implements Tokenizer {
        @Override
        public String regexPattern() {
            return null;
        }

        @Override
        public Map<String, Integer> getSpecialTokens() {
            return specialTokens;
        }

        @Override
        public boolean isSpecialToken(int tokenIndex) {
            return specialTokens.containsValue(tokenIndex);
        }

        @Override
        public boolean shouldDisplayToken(int token) {
            return !isSpecialToken(token);
        }

        @Override
        public List<Integer> encode(String text, Set<String> allowedSpecial) {
            throw new UnsupportedOperationException();
        }

        @Override
        public List<Integer> encodeAsList(String text) {
            throw new UnsupportedOperationException();
        }

        @Override
        public String decode(List<Integer> tokens) {
            throw new UnsupportedOperationException();
        }
    }
}
