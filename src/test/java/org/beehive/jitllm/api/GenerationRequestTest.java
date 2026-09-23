package org.beehive.jitllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;

public class GenerationRequestTest {

    @Test
    public void aPromptIsEnough() {
        GenerationRequest request = GenerationRequest.of("Why is the sky blue?");
        assertEquals("Why is the sky blue?", request.prompt());
        assertNull("no system prompt unless one is set", request.systemPrompt());
        assertTrue(request.maxNewTokens() > 0);
        assertEquals(List.of(), request.stopSequences());
    }

    /** A request with neither form is rejected at build time. */
    @Test
    public void aRequestWithNeitherFormIsRejectedAtBuildTime() {
        IllegalArgumentException thrown =
                assertThrows(
                        IllegalArgumentException.class, () -> GenerationRequest.builder().build());
        assertTrue(thrown.getMessage(), thrown.getMessage().contains("messages"));
    }

    @Test
    public void nonsensicalSamplingSettingsAreRejectedWhereTheyAreSet() {
        // Better a failure at build time than a sampler that silently does something else.
        assertThrows(
                IllegalArgumentException.class,
                () -> GenerationRequest.builder().prompt("x").maxNewTokens(0).build());
        assertThrows(
                IllegalArgumentException.class,
                () -> GenerationRequest.builder().prompt("x").temperature(-1).build());
        assertThrows(
                IllegalArgumentException.class,
                () -> GenerationRequest.builder().prompt("x").topP(0).build());
        assertThrows(
                IllegalArgumentException.class,
                () -> GenerationRequest.builder().prompt("x").topP(1.5f).build());
    }

    @Test
    public void aBuiltRequestDoesNotSeeLaterChangesToWhatItWasGiven() {
        List<String> stops = new ArrayList<>(List.of("</s>"));
        GenerationRequest request =
                GenerationRequest.builder().prompt("x").stopSequences(stops).build();
        stops.add("STOP");
        assertEquals(List.of("</s>"), request.stopSequences());
        assertThrows(
                UnsupportedOperationException.class, () -> request.stopSequences().add("STOP"));
    }

    @Test
    public void timingsDeriveRatesAndNeverDivideByZero() {
        GenerationTimings timings =
                new GenerationTimings(Duration.ofMillis(100), Duration.ofMillis(500), 21, 64);
        assertEquals(210.0, timings.promptTokensPerSecond(), 1e-6);
        assertEquals(128.0, timings.generatedTokensPerSecond(), 1e-6);

        GenerationTimings untimed = new GenerationTimings(Duration.ZERO, Duration.ZERO, 21, 64);
        assertEquals(
                "a run too short to time is not a run of infinite speed",
                0.0,
                untimed.generatedTokensPerSecond(),
                1e-9);
    }

    @Test
    public void aResultCarriesWhatItProducedAndWhyItStopped() {
        GenerationResult result =
                new GenerationResult(
                        "blue",
                        3,
                        1,
                        FinishReason.STOP_TOKEN,
                        new GenerationTimings(Duration.ofMillis(1), Duration.ofMillis(1), 3, 1));
        assertEquals("blue", result.text());
        assertEquals(FinishReason.STOP_TOKEN, result.finishReason());
        assertEquals(1, result.generatedTokens());
    }
}
