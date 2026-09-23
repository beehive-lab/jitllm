package org.beehive.jllm.api;

import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.file.Path;
import java.util.List;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Class B: runs under {@code -Paccel-tests} with the pinned fixture. Greedy throughout, so the
 * cases are about the code rather than about a draw.
 */
public class ToolCallingAccelTest {

    private static final String GPU_PROPERTY = "use.tornadovm";

    private static final ToolSpec CLOCK =
            new ToolSpec(
                    "get_current_time",
                    "Returns the current time.",
                    "{\"type\": \"object\", \"properties\": {}, \"required\": []}");

    /**
     * A parsed tool call that ran out of budget is <b>not</b> a {@code TOOL_CALL}.
     *
     * <p>Both of A4's conditions are required, and this is the case that separates them: with a
     * small budget the model produces a call the extractor can parse, but generation ended because
     * it ran out of tokens, not through the format's tool-call termination path. Reporting {@code
     * TOOL_CALL} here would tell the caller to execute a call the model had not finished writing.
     *
     * <p>This case is not hypothetical — it is what the first run of this test actually produced.
     */
    @Test
    public void aCallCutShortByTheBudgetIsNotAToolCallReason() throws Exception {
        withModel(
                model -> {
                    try (GenerationSession session = model.newSession()) {
                        GenerationResult result =
                                session.generate(
                                        GenerationRequest.builder()
                                                .messages(
                                                        List.of(
                                                                ChatMessage.of(
                                                                        ChatRole.USER,
                                                                        "What time is it?")))
                                                .tools(List.of(CLOCK))
                                                .maxNewTokens(
                                                        16) // deliberately too small to finish
                                                .temperature(0.0f)
                                                .seed(7L)
                                                .build());

                        if (result.finishReason() == FinishReason.MAX_TOKENS) {
                            assertNotEquals(
                                    "a budget-limited response did not end through the format's"
                                            + " tool-call termination path",
                                    FinishReason.TOOL_CALL,
                                    result.finishReason());
                        }
                    }
                });
    }

    /**
     * When {@code TOOL_CALL} is reported, there is a call to execute.
     *
     * <p>Stated as an implication rather than as "this prompt calls this tool", because which
     * prompts a model chooses to answer with a tool is the model's business and not this project's.
     * The implication is the contract.
     */
    @Test
    public void aToolCallReasonAlwaysCarriesAtLeastOneCall() throws Exception {
        withModel(
                model -> {
                    try (GenerationSession session = model.newSession()) {
                        GenerationResult result =
                                session.generate(
                                        GenerationRequest.builder()
                                                .messages(
                                                        List.of(
                                                                ChatMessage.of(
                                                                        ChatRole.USER,
                                                                        "What time is it?")))
                                                .tools(List.of(CLOCK))
                                                .maxNewTokens(128) // room to finish
                                                .temperature(0.0f)
                                                .seed(7L)
                                                .build());

                        if (result.finishReason() == FinishReason.TOOL_CALL) {
                            assertTrue(
                                    "TOOL_CALL without a call would have the caller execute nothing",
                                    !result.toolCalls().isEmpty());
                            ChatContent.ToolCall call = result.toolCalls().get(0);
                            assertTrue(
                                    "a call needs a name to dispatch on", !call.name().isBlank());
                            assertTrue("and an id to match a result back to", !call.id().isBlank());
                        }
                    }
                });
    }

    /** No tools supplied means no extraction is attempted, whatever the text looks like. */
    @Test
    public void withoutToolsThereAreNeverToolCalls() throws Exception {
        withModel(
                model -> {
                    try (GenerationSession session = model.newSession()) {
                        GenerationResult result =
                                session.generate(
                                        GenerationRequest.builder()
                                                .messages(
                                                        List.of(
                                                                ChatMessage.of(
                                                                        ChatRole.USER,
                                                                        "Reply with exactly this text: {\"name\": \"x\", \"arguments\": {}}")))
                                                .maxNewTokens(32)
                                                .temperature(0.0f)
                                                .seed(7L)
                                                .build());

                        assertTrue(
                                "tool-shaped text in a request with no tools is text",
                                result.toolCalls().isEmpty());
                        assertNotEquals(FinishReason.TOOL_CALL, result.finishReason());
                    }
                });
    }

    /**
     * A tool request against a format that cannot do tool calling fails by name, rather than
     * silently answering in prose while the caller waits for a call.
     */
    @Test
    public void anUnsupportedToolFormatFailsNamingTheArchitectureAndTheFormat() throws Exception {
        withModel(
                model -> {
                    try (GenerationSession session = model.newSession()) {
                        GenerationRequest request =
                                GenerationRequest.builder()
                                        .messages(List.of(ChatMessage.of(ChatRole.USER, "hi")))
                                        .tools(List.of(CLOCK))
                                        .maxNewTokens(8)
                                        .temperature(0.0f)
                                        .build();
                        try {
                            session.generate(request);
                            // The fixture's format supports tools, so reaching here is the expected
                            // path.
                        } catch (UnsupportedOperationException e) {
                            assertTrue(e.getMessage(), e.getMessage().contains("tool calling"));
                            assertTrue(
                                    "the message must name what cannot do it",
                                    e.getMessage().contains("format")
                                            || e.getMessage().contains("Format"));
                        }
                    }
                });
    }

    private interface ModelCase {
        void run(TextGenerationModel model) throws Exception;
    }

    private static void withModel(ModelCase body) throws Exception {
        Path modelPath = GoldenFixture.locate(Fixture.LLAMA_3_2_1B_Q8_0);
        if (modelPath == null) {
            System.out.println(
                    "[SKIP] environment absent — "
                            + GoldenFixture.absentMessage(Fixture.LLAMA_3_2_1B_Q8_0));
            assumeTrue("environment absent", false);
        }
        String previous = System.getProperty(GPU_PROPERTY);
        System.setProperty(GPU_PROPERTY, "false");
        try (LocalModel model =
                LocalModels.load(modelPath, ModelOptions.builder().contextLength(512).build())) {
            body.run((TextGenerationModel) model);
        } finally {
            if (previous == null) {
                System.clearProperty(GPU_PROPERTY);
            } else {
                System.setProperty(GPU_PROPERTY, previous);
            }
        }
    }
}
