package org.beehive.jitllm.api;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.beehive.jitllm.golden.GoldenFixture;
import org.junit.Test;

/**
 * Class B: tool calling across the families that report it, on the accelerator the SDK selects.
 *
 * <p>For each model file, greedy throughout, four exchanges in one session ({@code reset()} between
 * them, so one set of device buffers serves all four):
 *
 * <ol>
 *   <li><b>call</b> — {@code getWeather(city)} and "What is the weather in Paris?" must come back
 *       as {@code TOOL_CALL} with {@code getWeather {"city":"Paris"}};
 *   <li><b>answer</b> — that call and a result sent back must produce a text answer ({@code
 *       STOP_TOKEN}) that uses the result;
 *   <li><b>no-argument</b> — {@code get_current_time()} and "What time is it?";
 *   <li><b>two calls</b> — "… in Paris and in London?".
 * </ol>
 *
 * <p>Every model gets a {@code MATRIX} line with all four outcomes. The invariants of a reported
 * call — named, object arguments, distinct ids, and {@code TOOL_CALL} only with a call — are
 * asserted for every exchange. The first two outcomes are asserted for the models in {@link
 * #CONTRACT}, which pass them on CUDA and OpenCL; the rest are what the model chose, reported.
 *
 * <p>Models are file names under {@code $JITLLM_TEST_MODELS}. The default set is {@link #CONTRACT};
 * {@code -Djitllm.toolMatrix.models=a.gguf,b.gguf} replaces it. An absent file is skipped by name.
 * One model per JVM is the reliable way to run the large ones: a closed model's device memory is
 * not returned to the driver until the process exits.
 */
public class ToolCallingFamiliesAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";

    /** Models whose call and answer exchanges are asserted, not only reported. */
    static final List<String> CONTRACT =
            List.of(
                    "Qwen3-0.6B-Q8_0.gguf",
                    "qwen2.5-1.5b-instruct-q8_0.gguf",
                    "granite-3.2-2b-instruct-Q8_0.gguf",
                    "granite-4.0-1b-Q8_0.gguf");

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
                    "{\"type\":\"object\",\"properties\":{},\"required\":[]}");

    private static final String RESULT =
            "{\"temperature_celsius\":23,\"condition\":\"light rain\"}";

    @Test
    public void toolCapableFamiliesCallAndAnswer() throws Exception {
        String configured = System.getProperty("jitllm.toolMatrix.models");
        List<String> names =
                configured == null || configured.isBlank()
                        ? CONTRACT
                        : Arrays.stream(configured.split(",")).map(String::strip).toList();
        List<String> failures = new ArrayList<>();
        int ran = 0;
        for (String name : names) {
            Path path = GoldenFixture.modelsRoot().resolve(name);
            if (!Files.isRegularFile(path)) {
                System.out.println(
                        "[SKIP] " + name + " absent under " + GoldenFixture.modelsRoot());
                continue;
            }
            ran++;
            failures.addAll(exercise(name, path));
        }
        assumeTrue("no model present", ran > 0);
        assertTrue(
                "tool-calling contract failures:\n" + String.join("\n", failures),
                failures.isEmpty());
    }

    /** Runs the four exchanges; returns the contract failures. */
    private static List<String> exercise(String name, Path path) throws Exception {
        List<String> failures = new ArrayList<>();
        boolean contract = CONTRACT.contains(name);
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "true");
        try (LocalModel model =
                        LocalModels.load(path, ModelOptions.builder().contextLength(2048).build());
                GenerationSession session = ((TextGenerationModel) model).newSession()) {
            assertTrue(name + " reports tool calling", model.info().capabilities().toolCalling());
            String backend = session.prepare().backend();
            assertFalse(name + " runs on the accelerator", "CPU".equals(backend));
            StringBuilder matrix = new StringBuilder();

            // 1. call
            List<ChatMessage> conversation = new ArrayList<>();
            conversation.add(ChatMessage.of(ChatRole.USER, "What is the weather in Paris?"));
            GenerationResult call = session.generate(request(conversation, List.of(WEATHER)));
            print(name, "call", call);
            assertValid(name, call);
            boolean called =
                    call.finishReason() == FinishReason.TOOL_CALL
                            && call.toolCalls().get(0).name().equals("getWeather")
                            && call.toolCalls()
                                    .get(0)
                                    .argumentsJson()
                                    .replace(" ", "")
                                    .contains("\"city\":\"Paris\"");
            matrix.append(" call=").append(called ? "PASS" : "FAIL(" + summary(call) + ")");

            // 2. answer, from the call the test expected — sent even when the model did not make
            // it, so the answer exchange is exercised either way.
            ChatContent.ToolCall toolCall =
                    called
                            ? call.toolCalls().get(0)
                            : new ChatContent.ToolCall(
                                    "call-1", "getWeather", "{\"city\":\"Paris\"}");
            conversation.add(new ChatMessage(ChatRole.ASSISTANT, List.of(toolCall)));
            conversation.add(
                    new ChatMessage(
                            ChatRole.TOOL,
                            List.of(
                                    new ChatContent.ToolResult(
                                            toolCall.id(), toolCall.name(), RESULT))));
            session.reset();
            GenerationResult answer = session.generate(request(conversation, List.of(WEATHER)));
            print(name, "answer", answer);
            assertValid(name, answer);
            boolean answered =
                    answer.finishReason() == FinishReason.STOP_TOKEN
                            && answer.toolCalls().isEmpty()
                            && answer.text().contains("23");
            matrix.append(" answer=").append(answered ? "PASS" : "FAIL(" + summary(answer) + ")");

            // 3. no-argument call
            session.reset();
            GenerationResult clock =
                    session.generate(
                            request(
                                    List.of(ChatMessage.of(ChatRole.USER, "What time is it?")),
                                    List.of(CLOCK, WEATHER)));
            print(name, "no-arg", clock);
            assertValid(name, clock);
            boolean noArg =
                    clock.finishReason() == FinishReason.TOOL_CALL
                            && clock.toolCalls().size() == 1
                            && clock.toolCalls().get(0).name().equals("get_current_time")
                            && clock.toolCalls()
                                    .get(0)
                                    .argumentsJson()
                                    .replace(" ", "")
                                    .equals("{}");
            matrix.append(" no-arg=").append(noArg ? "PASS" : "MODEL(" + summary(clock) + ")");

            // 4. two calls
            session.reset();
            GenerationResult both =
                    session.generate(
                            request(
                                    List.of(
                                            ChatMessage.of(
                                                    ChatRole.USER,
                                                    "What is the weather in Paris and in London?")),
                                    List.of(WEATHER)));
            print(name, "two", both);
            assertValid(name, both);
            String args =
                    both.toolCalls().stream()
                            .map(ChatContent.ToolCall::argumentsJson)
                            .reduce("", String::concat);
            boolean two =
                    both.finishReason() == FinishReason.TOOL_CALL
                            && both.toolCalls().size() == 2
                            && args.contains("Paris")
                            && args.contains("London");
            matrix.append(" two-calls=").append(two ? "PASS" : "MODEL(" + summary(both) + ")");

            System.out.println("MATRIX " + name + " backend=" + backend + matrix);
            if (contract && !called) {
                failures.add(name + " call: " + call.finishReason() + " " + call.text());
            }
            if (contract && !answered) {
                failures.add(name + " answer: " + answer.finishReason() + " " + answer.text());
            }
        } finally {
            if (previous == null) {
                System.clearProperty(GPU_PROPERTY);
            } else {
                System.setProperty(GPU_PROPERTY, previous);
            }
        }
        return failures;
    }

    /** The invariants of any reported call, whatever the model chose. */
    private static void assertValid(String name, GenerationResult result) {
        if (result.finishReason() == FinishReason.TOOL_CALL) {
            assertFalse(name + ": TOOL_CALL carries calls", result.toolCalls().isEmpty());
        }
        List<String> ids = new ArrayList<>();
        for (ChatContent.ToolCall call : result.toolCalls()) {
            assertFalse(name + ": a call is named", call.name().isBlank());
            assertTrue(
                    name + ": arguments are an object: " + call.argumentsJson(),
                    call.argumentsJson().strip().startsWith("{"));
            assertFalse(name + ": ids are distinct", ids.contains(call.id()));
            ids.add(call.id());
        }
    }

    private static void print(String name, String exchange, GenerationResult result) {
        System.out.println(
                "["
                        + name
                        + "] "
                        + exchange
                        + ": "
                        + result.finishReason()
                        + " "
                        + result.text().replace("\n", "\\n"));
    }

    private static String summary(GenerationResult result) {
        return result.finishReason() + ", " + result.toolCalls().size() + " call(s)";
    }

    private static GenerationRequest request(List<ChatMessage> conversation, List<ToolSpec> tools) {
        return GenerationRequest.builder()
                .messages(List.copyOf(conversation))
                .tools(tools)
                .maxNewTokens(512)
                .temperature(0.0f)
                .seed(42L)
                .build();
    }
}
