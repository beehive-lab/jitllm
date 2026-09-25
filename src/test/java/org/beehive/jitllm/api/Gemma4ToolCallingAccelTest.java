package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.golden.GoldenFixture;
import org.beehive.jitllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Class B: Gemma 4 tool calling end to end on the accelerator, for both pinned quantizations.
 *
 * <p>Runs under {@code -Paccel-tests} with {@code use.tornadovm=true}, skipping explicitly when a
 * fixture is absent. Gemma 4's FP16 key/value cache, the default, is verified on CUDA, which is the
 * backend the profile pins.
 *
 * <p>One exchange per fixture, greedy: a {@code getWeather(city)} tool and the question "What is
 * the weather in Paris?" must come back as a parsed call with {@link FinishReason#TOOL_CALL}; the
 * call and a tool result are then sent back as history, and the model must answer in text that uses
 * the result. The rendering itself is pinned against the template by {@code
 * Gemma4ToolConversationTest}; this is the check that the model reads it the way it was trained to.
 */
public class Gemma4ToolCallingAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";

    private static final ToolSpec WEATHER =
            new ToolSpec(
                    "getWeather",
                    "Returns the current weather for a city.",
                    "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\","
                            + "\"description\":\"The city name\"}},\"required\":[\"city\"]}");

    private static final String QUESTION = "What is the weather in Paris?";

    /** A distinctive number, so an answer that uses the result is recognisable as doing so. */
    private static final String RESULT =
            "{\"temperature_celsius\":23,\"condition\":\"light rain\"}";

    @Test
    public void q8_0CallsTheToolAndAnswersFromTheResult() throws Exception {
        callAndAnswer(Fixture.GEMMA_4_E2B_Q8_0);
    }

    @Test
    public void q4_0CallsTheToolAndAnswersFromTheResult() throws Exception {
        callAndAnswer(Fixture.GEMMA_4_E2B_Q4_0);
    }

    private static void callAndAnswer(Fixture fixture) throws Exception {
        Path modelPath = GoldenFixture.locate(fixture);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — " + GoldenFixture.absentMessage(fixture));
            assumeTrue("environment absent", false);
        }
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        try (LocalModel model =
                LocalModels.load(modelPath, ModelOptions.builder().contextLength(1024).build())) {
            assertTrue(
                    fixture + ": the file's vocabulary carries the tool markers",
                    model.info().capabilities().toolCalling());

            List<ChatMessage> conversation = new ArrayList<>();
            conversation.add(ChatMessage.of(ChatRole.USER, QUESTION));
            try (GenerationSession session = ((TextGenerationModel) model).newSession()) {
                var execution = session.prepare();
                System.out.println("[" + fixture + "] execution: " + execution);
                assertTrue(
                        fixture + ": runs on the accelerator, not the CPU: " + execution,
                        !"CPU".equals(execution.backend()));

                GenerationResult call = session.generate(request(conversation));
                System.out.println(
                        "[" + fixture + "] call turn: " + call.finishReason() + " " + call.text());

                assertEquals(
                        fixture
                                + ": the model ends a call through the tool-call path, got text: "
                                + call.text(),
                        FinishReason.TOOL_CALL,
                        call.finishReason());
                assertEquals(fixture + ": one call", 1, call.toolCalls().size());
                ChatContent.ToolCall toolCall = call.toolCalls().get(0);
                assertEquals("getWeather", toolCall.name());
                assertTrue(
                        fixture + ": the argument names the city: " + toolCall.argumentsJson(),
                        toolCall.argumentsJson().replace(" ", "").contains("\"city\":\"Paris\""));

                conversation.add(new ChatMessage(ChatRole.ASSISTANT, List.of(toolCall)));
                conversation.add(
                        new ChatMessage(
                                ChatRole.TOOL,
                                List.of(
                                        new ChatContent.ToolResult(
                                                toolCall.id(), toolCall.name(), RESULT))));

                GenerationResult answer = session.generate(request(conversation));
                System.out.println(
                        "["
                                + fixture
                                + "] answer turn: "
                                + answer.finishReason()
                                + " "
                                + answer.text());

                assertEquals(
                        fixture + ": the answer is text, ended by the model: " + answer.text(),
                        FinishReason.STOP_TOKEN,
                        answer.finishReason());
                assertTrue(
                        fixture + ": the answer calls no further tool",
                        answer.toolCalls().isEmpty());
                String text = answer.text().toLowerCase();
                assertTrue(
                        fixture + ": the answer uses the tool result: " + answer.text(),
                        text.contains("23") && text.contains("rain"));
            }
        } finally {
            if (previous == null) {
                System.clearProperty(GPU_PROPERTY);
            } else {
                System.setProperty(GPU_PROPERTY, previous);
            }
        }
    }

    private static final ToolSpec CLOCK =
            new ToolSpec(
                    "get_current_time",
                    "Returns the current time.",
                    "{\"type\": \"object\", \"properties\": {}, \"required\": []}");

    @Test
    public void q8_0NoArgumentParallelAndStreamedCalls() throws Exception {
        otherCallShapes(Fixture.GEMMA_4_E2B_Q8_0);
    }

    @Test
    public void q4_0NoArgumentParallelAndStreamedCalls() throws Exception {
        otherCallShapes(Fixture.GEMMA_4_E2B_Q4_0);
    }

    /**
     * The other shapes an integration relies on: a call with no arguments, two calls in one turn,
     * and a streamed call, whose events concatenate to the text the call was extracted from.
     */
    private static void otherCallShapes(Fixture fixture) throws Exception {
        Path modelPath = GoldenFixture.locate(fixture);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — " + GoldenFixture.absentMessage(fixture));
            assumeTrue("environment absent", false);
        }
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        try (LocalModel model =
                LocalModels.load(modelPath, ModelOptions.builder().contextLength(1024).build())) {
            TextGenerationModel generator = (TextGenerationModel) model;

            try (GenerationSession session = generator.newSession()) {
                GenerationResult clock =
                        session.generate(
                                request(
                                        List.of(ChatMessage.of(ChatRole.USER, "What time is it?")),
                                        List.of(CLOCK, WEATHER)));
                System.out.println("[" + fixture + "] no-argument: " + clock.text());
                assertEquals(clock.text(), FinishReason.TOOL_CALL, clock.finishReason());
                assertEquals(1, clock.toolCalls().size());
                assertEquals("get_current_time", clock.toolCalls().get(0).name());
                assertEquals("{}", clock.toolCalls().get(0).argumentsJson());
            }

            try (GenerationSession session = generator.newSession()) {
                GenerationResult both =
                        session.generate(
                                request(
                                        List.of(
                                                ChatMessage.of(
                                                        ChatRole.USER,
                                                        "What is the weather in Paris and in"
                                                                + " London?")),
                                        List.of(WEATHER)));
                System.out.println("[" + fixture + "] two calls: " + both.text());
                assertEquals(both.text(), FinishReason.TOOL_CALL, both.finishReason());
                assertEquals(both.text(), 2, both.toolCalls().size());
                assertTrue(
                        "the model ends a run of calls with <eos>, which stops it: " + both.text(),
                        !both.text().contains("<eos>"));
                assertEquals(
                        List.of("getWeather", "getWeather"),
                        both.toolCalls().stream().map(ChatContent.ToolCall::name).toList());
                String args =
                        both.toolCalls().get(0).argumentsJson()
                                + both.toolCalls().get(1).argumentsJson();
                assertTrue(args, args.contains("Paris") && args.contains("London"));
                assertTrue(
                        "two calls get two ids",
                        !both.toolCalls().get(0).id().equals(both.toolCalls().get(1).id()));
            }

            try (GenerationSession session = generator.newSession()) {
                StringBuilder streamed = new StringBuilder();
                GenerationResult result =
                        session.generate(
                                GenerationRequest.builder()
                                        .messages(List.of(ChatMessage.of(ChatRole.USER, QUESTION)))
                                        .tools(List.of(WEATHER))
                                        .maxNewTokens(160)
                                        .temperature(0.0f)
                                        .seed(42L)
                                        .onToken(streamed::append)
                                        .build());
                System.out.println("[" + fixture + "] streamed: " + streamed);
                assertEquals(FinishReason.TOOL_CALL, result.finishReason());
                assertEquals(
                        "the stream is the text the call was read from",
                        result.text(),
                        streamed.toString());
                assertEquals("getWeather", result.toolCalls().get(0).name());
            }
        } finally {
            if (previous == null) {
                System.clearProperty(GPU_PROPERTY);
            } else {
                System.setProperty(GPU_PROPERTY, previous);
            }
        }
    }

    private static GenerationRequest request(List<ChatMessage> conversation) {
        return request(conversation, List.of(WEATHER));
    }

    private static GenerationRequest request(List<ChatMessage> conversation, List<ToolSpec> tools) {
        return GenerationRequest.builder()
                .messages(List.copyOf(conversation))
                .tools(tools)
                .maxNewTokens(160)
                .temperature(0.0f)
                .seed(42L)
                .build();
    }
}
