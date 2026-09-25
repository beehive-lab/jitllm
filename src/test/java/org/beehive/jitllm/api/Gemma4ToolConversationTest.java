package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jitllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jitllm.inference.sampler.Sampler;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.Configuration;
import org.beehive.jitllm.model.Model;
import org.beehive.jitllm.model.ModelType;
import org.beehive.jitllm.model.format.ChatFormat;
import org.beehive.jitllm.model.format.Gemma4ChatFormat;
import org.beehive.jitllm.model.format.Gemma4TestVocabulary;
import org.beehive.jitllm.tokenizer.Gemma4Tokenizer;
import org.beehive.jitllm.tokenizer.Tokenizer;
import org.junit.Test;

/**
 * Whole Gemma 4 conversations with tools, through the facade's encoder, against the reference
 * template.
 *
 * <p>Each expected string is the GGUF-embedded chat template of {@code gemma-4-E2B-it-Q8_0.gguf} /
 * {@code gemma-4-E2B-it-Q4_0.gguf} rendered with Jinja2 (Hugging Face settings, {@code
 * add_generation_prompt=True}) for the same messages and tools, the tools given as the OpenAI-style
 * objects the facade builds and the tool results as {@code role: tool} messages with string
 * content. The synthetic vocabulary decodes to exactly what was encoded, so a decoded encoding
 * equal to the template's text is the same prompt; the marker tokens are then checked to be single
 * ids.
 */
public class Gemma4ToolConversationTest {

    private static final String Q = "<|\"|>";

    private static final ToolSpec WEATHER =
            new ToolSpec(
                    "getWeather",
                    "Returns the current weather for a city.",
                    "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\","
                            + "\"description\":\"The city name\"}},\"required\":[\"city\"]}");

    private static final ToolSpec CLOCK =
            new ToolSpec(
                    "get_current_time",
                    "Returns the current time.",
                    "{\"type\": \"object\", \"properties\": {}, \"required\": []}");

    private static final String WEATHER_DECLARATION =
            "<|tool>declaration:getWeather{description:<|\"|>Returns the current weather for a"
                    + " city.<|\"|>,parameters:{properties:{city:{description:<|\"|>The city"
                    + " name<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:"
                    + "<|\"|>OBJECT<|\"|>}}<tool|>";

    private static final String CLOCK_DECLARATION =
            "<|tool>declaration:get_current_time{description:<|\"|>Returns the current time."
                    + "<|\"|>,parameters:{type:<|\"|>OBJECT<|\"|>}}<tool|>";

    private final Gemma4Tokenizer tokenizer = Gemma4TestVocabulary.tokenizer();
    private final ConversationEncoder encoder =
            new ConversationEncoder(
                    new FormatOnlyModel(new Gemma4ChatFormat(tokenizer)), ThinkingMode.DEFAULT);

    @Test
    public void withoutToolsNothingChanges() {
        assertEncodes(
                "<bos><|turn>user\nHi<turn|>\n<|turn>model\n",
                List.of(ChatMessage.of(ChatRole.USER, "Hi")),
                List.of());
    }

    /** The declarations follow the system text inside the system turn, with nothing between. */
    @Test
    public void toolsGoAtTheEndOfTheSystemTurn() {
        assertEncodes(
                "<bos><|turn>system\nYou are helpful."
                        + WEATHER_DECLARATION
                        + "<turn|>\n<|turn>user\nWhat is the weather in Paris?<turn|>\n"
                        + "<|turn>model\n",
                List.of(
                        ChatMessage.of(ChatRole.SYSTEM, "You are helpful."),
                        ChatMessage.of(ChatRole.USER, "What is the weather in Paris?")),
                List.of(WEATHER));
    }

    /** No system message: the template opens a system turn for the declarations alone. */
    @Test
    public void withoutASystemMessageTheToolsGetASystemTurnOfTheirOwn() {
        assertEncodes(
                "<bos><|turn>system\n"
                        + WEATHER_DECLARATION
                        + CLOCK_DECLARATION
                        + "<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n",
                List.of(ChatMessage.of(ChatRole.USER, "Hi")),
                List.of(WEATHER, CLOCK));
    }

    /**
     * The result is written into the turn that made the call, and the prompt ends there: the
     * model's answer continues the same turn, so there is no new {@code <|turn>model\n}.
     */
    @Test
    public void aToolResultStaysInTheCallingTurnAndTheAnswerContinuesIt() {
        assertEncodes(
                "<bos><|turn>system\n"
                        + WEATHER_DECLARATION
                        + "<turn|>\n<|turn>user\nWhat is the weather in Paris?<turn|>\n"
                        + "<|turn>model\n<|tool_call>call:getWeather{city:"
                        + Q
                        + "Paris"
                        + Q
                        + "}<tool_call|><|tool_response>response:getWeather{value:"
                        + Q
                        + "{\"temperature\":18,\"condition\":\"sunny\"}"
                        + Q
                        + "}<tool_response|>",
                List.of(
                        ChatMessage.of(ChatRole.USER, "What is the weather in Paris?"),
                        new ChatMessage(
                                ChatRole.ASSISTANT,
                                List.of(
                                        new ChatContent.ToolCall(
                                                "c1", "getWeather", "{\"city\":\"Paris\"}"))),
                        new ChatMessage(
                                ChatRole.TOOL,
                                List.of(
                                        new ChatContent.ToolResult(
                                                "c1",
                                                "getWeather",
                                                "{\"temperature\":18,\"condition\":\"sunny\"}")))),
                List.of(WEATHER));
    }

    /**
     * Two calls in one turn, both results, the answer in the same turn, and a new user turn: the
     * full shape of a finished tool exchange in history.
     */
    @Test
    public void twoCallsTheirResultsTheAnswerAndTheNextUserTurn() {
        assertEncodes(
                "<bos><|turn>system\n"
                        + WEATHER_DECLARATION
                        + CLOCK_DECLARATION
                        + "<turn|>\n<|turn>user\nWeather in Paris and the time?<turn|>\n"
                        + "<|turn>model\n<|tool_call>call:getWeather{city:"
                        + Q
                        + "Paris"
                        + Q
                        + "}<tool_call|><|tool_call>call:get_current_time{}<tool_call|>"
                        + "<|tool_response>response:getWeather{value:"
                        + Q
                        + "sunny"
                        + Q
                        + "}<tool_response|><|tool_response>response:get_current_time{value:"
                        + Q
                        + "12:00"
                        + Q
                        + "}<tool_response|>It is sunny and noon.<turn|>\n"
                        + "<|turn>user\nThanks<turn|>\n<|turn>model\n",
                List.of(
                        ChatMessage.of(ChatRole.USER, "Weather in Paris and the time?"),
                        new ChatMessage(
                                ChatRole.ASSISTANT,
                                List.of(
                                        new ChatContent.ToolCall(
                                                "c1", "getWeather", "{\"city\":\"Paris\"}"),
                                        new ChatContent.ToolCall("c2", "get_current_time", "{}"))),
                        new ChatMessage(
                                ChatRole.TOOL,
                                List.of(new ChatContent.ToolResult("c1", "getWeather", "sunny"))),
                        new ChatMessage(
                                ChatRole.TOOL,
                                List.of(
                                        new ChatContent.ToolResult(
                                                "c2", "get_current_time", "12:00"))),
                        ChatMessage.of(ChatRole.ASSISTANT, "It is sunny and noon."),
                        ChatMessage.of(ChatRole.USER, "Thanks")),
                List.of(WEATHER, CLOCK));
    }

    /** A user turn straight after the results closes the open model turn first. */
    @Test
    public void aUserTurnAfterTheResultsClosesTheModelTurn() {
        assertEncodes(
                "<bos><|turn>system\n"
                        + CLOCK_DECLARATION
                        + "<turn|>\n<|turn>user\nTime?<turn|>\n<|turn>model\n"
                        + "<|tool_call>call:get_current_time{}<tool_call|>"
                        + "<|tool_response>response:get_current_time{value:"
                        + Q
                        + "12:00"
                        + Q
                        + "}<tool_response|><turn|>\n<|turn>user\nAnd now?<turn|>\n"
                        + "<|turn>model\n",
                List.of(
                        ChatMessage.of(ChatRole.USER, "Time?"),
                        new ChatMessage(
                                ChatRole.ASSISTANT,
                                List.of(new ChatContent.ToolCall("c1", "get_current_time", "{}"))),
                        new ChatMessage(
                                ChatRole.TOOL,
                                List.of(
                                        new ChatContent.ToolResult(
                                                "c1", "get_current_time", "12:00"))),
                        ChatMessage.of(ChatRole.USER, "And now?")),
                List.of(CLOCK));
    }

    /** The markers are the vocabulary's single tokens, not their spelling in characters. */
    @Test
    public void theMarkersAreEncodedAsSingleTokens() {
        List<Integer> tokens =
                encoder.encode(
                        List.of(
                                ChatMessage.of(ChatRole.USER, "Paris?"),
                                new ChatMessage(
                                        ChatRole.ASSISTANT,
                                        List.of(
                                                new ChatContent.ToolCall(
                                                        "c1",
                                                        "getWeather",
                                                        "{\"city\":\"Paris\"}"))),
                                new ChatMessage(
                                        ChatRole.TOOL,
                                        List.of(
                                                new ChatContent.ToolResult(
                                                        "c1", "getWeather", "sunny")))),
                        List.of(WEATHER));
        var special = tokenizer.getSpecialTokens();
        assertEquals(1, Collections.frequency(tokens, special.get("<|tool>")));
        assertEquals(1, Collections.frequency(tokens, special.get("<tool|>")));
        assertEquals(1, Collections.frequency(tokens, special.get("<|tool_call>")));
        assertEquals(1, Collections.frequency(tokens, special.get("<tool_call|>")));
        assertEquals(1, Collections.frequency(tokens, special.get("<|tool_response>")));
        assertEquals(1, Collections.frequency(tokens, special.get("<tool_response|>")));
        // description, property description, STRING, city, OBJECT, Paris, sunny: 7 strings
        assertEquals(14, Collections.frequency(tokens, special.get(Q)));
    }

    @Test
    public void callsAreExtractedWithPositionalIds() {
        assertEquals(List.of(), encoder.extractToolCalls("It is sunny."));
        assertEquals(
                List.of(new ChatContent.ToolCall("call-1", "getWeather", "{\"city\":\"Paris\"}")),
                encoder.extractToolCalls(
                        "<|tool_call>call:getWeather{city:" + Q + "Paris" + Q + "}<tool_call|>"));
        assertEquals(
                List.of(
                        new ChatContent.ToolCall("call-1", "getWeather", "{\"city\":\"Paris\"}"),
                        new ChatContent.ToolCall("call-2", "get_current_time", "{}")),
                encoder.extractToolCalls(
                        "<|tool_call>call:getWeather{city:"
                                + Q
                                + "Paris"
                                + Q
                                + "}<tool_call|><|tool_call>call:get_current_time{}<tool_call|>"));
    }

    @Test
    public void aToolRequestUsesTheToolAwareStopTokens() {
        var special = tokenizer.getSpecialTokens();
        assertEquals(Set.of(special.get("<turn|>")), encoder.stopTokens(false));
        assertEquals(
                Set.of(
                        special.get("<turn|>"),
                        special.get("<|tool_response>"),
                        tokenizer.tokenIndex("<eos>")),
                encoder.stopTokens(true));
        assertTrue(encoder.supportsTools());
    }

    private void assertEncodes(String expected, List<ChatMessage> messages, List<ToolSpec> tools) {
        assertEquals(expected, tokenizer.decode(encoder.encode(messages, tools)));
    }

    /** A model that is only its chat format: the encoder reads nothing else. */
    static final class FormatOnlyModel implements Model {

        private final ChatFormat chatFormat;

        private final boolean beginOfText;

        FormatOnlyModel(ChatFormat chatFormat) {
            this(chatFormat, true);
        }

        /**
         * @param beginOfText whether the family's model starts a conversation with the format's
         *     begin-of-text token, as {@code Model.shouldAddBeginOfText()} says (Qwen's do not)
         */
        FormatOnlyModel(ChatFormat chatFormat, boolean beginOfText) {
            this.chatFormat = chatFormat;
            this.beginOfText = beginOfText;
        }

        @Override
        public boolean shouldAddBeginOfText() {
            return beginOfText;
        }

        @Override
        public ChatFormat chatFormat() {
            return chatFormat;
        }

        @Override
        public ModelType getModelType() {
            return ModelType.GEMMA_4;
        }

        @Override
        public Configuration configuration() {
            throw notNeeded();
        }

        @Override
        public Tokenizer tokenizer() {
            throw notNeeded();
        }

        @Override
        public Weights weights() {
            throw notNeeded();
        }

        @Override
        public State createNewState() {
            throw notNeeded();
        }

        @Override
        public State createNewState(int batchsize) {
            throw notNeeded();
        }

        @Override
        public List<Integer> generateTokens(
                State state,
                int startPosition,
                List<Integer> promptTokens,
                Set<Integer> stopTokens,
                int maxTokens,
                Sampler sampler,
                boolean echo,
                IntConsumer onTokenGenerated) {
            throw notNeeded();
        }

        @Override
        public List<Integer> generateTokensGPU(
                State state,
                int startPosition,
                List<Integer> promptTokens,
                Set<Integer> stopTokens,
                int maxTokens,
                Sampler sampler,
                boolean echo,
                IntConsumer onTokenGenerated,
                TornadoVMMasterPlan plan) {
            throw notNeeded();
        }

        private static UnsupportedOperationException notNeeded() {
            return new UnsupportedOperationException("the encoder only reads the chat format");
        }
    }
}
