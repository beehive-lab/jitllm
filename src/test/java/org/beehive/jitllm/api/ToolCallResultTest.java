package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;

/**
 * Class A: the result type's own contract. The end-to-end cases — a real model asking for a real
 * tool, and a model producing tool-shaped text that does not parse — are in {@code
 * ToolCallingAccelTest}.
 */
public class ToolCallResultTest {

    private static final GenerationTimings TIMINGS =
            new GenerationTimings(Duration.ZERO, Duration.ZERO, 1, 1);

    @Test
    public void aResultWithoutToolCallsHasAnEmptyImmutableList() {
        GenerationResult result =
                new GenerationResult("hello", 1, 1, FinishReason.STOP_TOKEN, TIMINGS);
        assertTrue(result.toolCalls().isEmpty());
        assertThrows(
                UnsupportedOperationException.class,
                () -> result.toolCalls().add(new ChatContent.ToolCall("id", "t", "{}")));
    }

    @Test
    public void toolCallsAreCopiedSoTheCallerCannotMutateTheResult() {
        List<ChatContent.ToolCall> mutable = new ArrayList<>();
        mutable.add(new ChatContent.ToolCall("id-1", "clock", "{}"));
        GenerationResult result =
                new GenerationResult("", 1, 1, FinishReason.TOOL_CALL, TIMINGS, mutable);
        mutable.clear();
        assertEquals("the result kept the calls it was built with", 1, result.toolCalls().size());
    }

    /**
     * TOOL_CALL is a stop reason, and the enum keeps the four that existed. A fifth constant
     * appearing silently would be a change to what callers must handle.
     */
    @Test
    public void theStopReasonsAreTheFourThatExistedPlusToolCall() {
        assertEquals(
                java.util.EnumSet.of(
                        FinishReason.STOP_TOKEN,
                        FinishReason.MAX_TOKENS,
                        FinishReason.STOP_SEQUENCE,
                        FinishReason.CONTEXT_FULL,
                        FinishReason.TOOL_CALL),
                java.util.EnumSet.allOf(FinishReason.class));
    }
}
