package org.beehive.jitllm.server;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import org.junit.Test;

/**
 * Token usage on a streamed reply. OpenAI clients (langchain4j among them) ask for it with {@code
 * stream_options.include_usage} and read it from one extra chunk, sent just before {@code [DONE]},
 * whose {@code choices} is empty. Without it a streaming client has no in/out token counts at all.
 */
public class OpenAIServerStreamUsageTest {

    // ---- includeUsage ---------------------------------------------------------------------

    @Test
    public void usageIsRequestedViaStreamOptions() {
        Map<String, Object> body =
                Json.parseObject("{\"stream\":true,\"stream_options\":{\"include_usage\":true}}");

        assertTrue(OpenAIServer.includeUsage(body));
    }

    @Test
    public void noStreamOptionsMeansNoUsageChunk() {
        assertFalse(OpenAIServer.includeUsage(Json.parseObject("{\"stream\":true}")));
    }

    @Test
    public void anExplicitFalseMeansNoUsageChunk() {
        Map<String, Object> body =
                Json.parseObject("{\"stream\":true,\"stream_options\":{\"include_usage\":false}}");

        assertFalse(OpenAIServer.includeUsage(body));
    }

    /** A malformed {@code stream_options} must not fail the request — usage is optional. */
    @Test
    public void aNonObjectStreamOptionsIsIgnored() {
        assertFalse(OpenAIServer.includeUsage(Json.parseObject("{\"stream_options\":true}")));
    }

    // ---- usageChunk -----------------------------------------------------------------------

    @Test
    @SuppressWarnings("unchecked")
    public void theUsageChunkHasEmptyChoicesAndTheCounts() {
        Map<String, Object> chunk =
                OpenAIServer.usageChunk(
                        "chatcmpl-1", "chat.completion.chunk", 1790854405L, "gemma", 15, 3);

        assertEquals("chatcmpl-1", chunk.get("id"));
        assertEquals("chat.completion.chunk", chunk.get("object"));
        assertEquals("gemma", chunk.get("model"));
        assertEquals(List.of(), chunk.get("choices"));
        Map<String, Object> usage = (Map<String, Object>) chunk.get("usage");
        assertEquals(15, usage.get("prompt_tokens"));
        assertEquals(3, usage.get("completion_tokens"));
        assertEquals(18, usage.get("total_tokens"));
    }

    // ---- the SSE stream itself, over HTTP, with a stub generator ----------------------------

    private static final int PROMPT_TOKENS = 15;
    private static final int COMPLETION_TOKENS = 2;

    /** Streams "Hel" then "lo" and reports fixed counts, so the wire format needs no model. */
    private static final class StubGenerator implements OpenAIServer.Generator {
        @Override
        public InferenceService.Result generate(
                InferenceService.Request request, Consumer<String> onToken) {
            onToken.accept("Hel");
            onToken.accept("lo");
            return new InferenceService.Result("Hello", PROMPT_TOKENS, COMPLETION_TOKENS, true);
        }

        @Override
        public void close() {}
    }

    private static final String CHAT = "/v1/chat/completions";
    private static final String COMPLETIONS = "/v1/completions";

    private static String requestBody(String endpoint, String streamOptions) {
        String input =
                endpoint.equals(CHAT)
                        ? "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]"
                        : "\"prompt\":\"Hi\"";
        return "{" + input + ",\"stream\":true" + streamOptions + "}";
    }

    /**
     * POSTs {@code body} to {@code endpoint} and returns the SSE data payloads in order; the
     * terminating {@code [DONE]} is kept as the literal string, every other event is parsed.
     */
    private static List<Object> stream(String endpoint, String body) throws Exception {
        try (OpenAIServer server =
                new OpenAIServer(
                        new StubGenerator(), "stub", false, OpenAIServer.UNKNOWN_CONTEXT_LENGTH)) {
            server.start("127.0.0.1", 0);
            HttpResponse<String> response =
                    HttpClient.newHttpClient()
                            .send(
                                    HttpRequest.newBuilder(
                                                    URI.create(
                                                            "http://127.0.0.1:"
                                                                    + server.port()
                                                                    + endpoint))
                                            .timeout(Duration.ofSeconds(10))
                                            .POST(HttpRequest.BodyPublishers.ofString(body))
                                            .build(),
                                    HttpResponse.BodyHandlers.ofString());
            assertEquals(response.body(), 200, response.statusCode());
            List<Object> events = new ArrayList<>();
            for (String event : response.body().split("\n\n")) {
                assertTrue(event, event.startsWith("data: "));
                String data = event.substring("data: ".length());
                events.add(data.equals("[DONE]") ? data : Json.parseObject(data));
            }
            return events;
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> onlyChoice(Map<String, Object> chunk) {
        List<Object> choices = (List<Object>) chunk.get("choices");
        assertEquals(chunk.toString(), 1, choices.size());
        return (Map<String, Object>) choices.get(0);
    }

    /**
     * With {@code include_usage}: every ordinary chunk carries {@code "usage": null}, the finish
     * chunk is followed by exactly one usage chunk with empty choices and the counts, then {@code
     * [DONE]}.
     */
    @SuppressWarnings("unchecked")
    private static void assertUsageStream(String endpoint) throws Exception {
        List<Object> events =
                stream(
                        endpoint,
                        requestBody(endpoint, ",\"stream_options\":{\"include_usage\":true}"));

        int n = events.size();
        assertEquals("[DONE]", events.get(n - 1));
        Map<String, Object> usageChunk = (Map<String, Object>) events.get(n - 2);
        Map<String, Object> finishChunk = (Map<String, Object>) events.get(n - 3);

        assertEquals("stop", onlyChoice(finishChunk).get("finish_reason"));
        for (Object event : events.subList(0, n - 2)) {
            Map<String, Object> chunk = (Map<String, Object>) event;
            assertTrue(chunk.toString(), chunk.containsKey("usage"));
            assertNull(chunk.toString(), chunk.get("usage"));
            assertEquals(1, ((List<Object>) chunk.get("choices")).size());
        }

        assertEquals(List.of(), usageChunk.get("choices"));
        assertEquals(finishChunk.get("id"), usageChunk.get("id"));
        assertEquals(finishChunk.get("object"), usageChunk.get("object"));
        assertEquals("stub", usageChunk.get("model"));
        Map<String, Object> usage = (Map<String, Object>) usageChunk.get("usage");
        assertEquals(PROMPT_TOKENS, ((Number) usage.get("prompt_tokens")).intValue());
        assertEquals(COMPLETION_TOKENS, ((Number) usage.get("completion_tokens")).intValue());
        assertEquals(
                PROMPT_TOKENS + COMPLETION_TOKENS, ((Number) usage.get("total_tokens")).intValue());
    }

    /** Without {@code include_usage}: no chunk mentions usage, and finish is followed by [DONE]. */
    @SuppressWarnings("unchecked")
    private static void assertNoUsageStream(String endpoint, String streamOptions)
            throws Exception {
        List<Object> events = stream(endpoint, requestBody(endpoint, streamOptions));

        int n = events.size();
        assertEquals("[DONE]", events.get(n - 1));
        assertEquals(
                "stop", onlyChoice((Map<String, Object>) events.get(n - 2)).get("finish_reason"));
        for (Object event : events.subList(0, n - 1)) {
            Map<String, Object> chunk = (Map<String, Object>) event;
            assertFalse(chunk.toString(), chunk.containsKey("usage"));
            assertEquals(1, ((List<Object>) chunk.get("choices")).size());
        }
    }

    @Test
    public void chatStreamEndsWithFinishThenUsageThenDone() throws Exception {
        assertUsageStream(CHAT);
    }

    @Test
    public void completionsStreamEndsWithFinishThenUsageThenDone() throws Exception {
        assertUsageStream(COMPLETIONS);
    }

    @Test
    public void chatStreamHasNoUsageWhenNotAskedFor() throws Exception {
        assertNoUsageStream(CHAT, "");
        assertNoUsageStream(CHAT, ",\"stream_options\":{\"include_usage\":false}");
    }

    @Test
    public void completionsStreamHasNoUsageWhenNotAskedFor() throws Exception {
        assertNoUsageStream(COMPLETIONS, "");
        assertNoUsageStream(COMPLETIONS, ",\"stream_options\":{\"include_usage\":false}");
    }

    /** The streamed text itself is unchanged by asking for usage. */
    @Test
    @SuppressWarnings("unchecked")
    public void chatStreamCarriesRoleThenContent() throws Exception {
        List<Object> events =
                stream(CHAT, requestBody(CHAT, ",\"stream_options\":{\"include_usage\":true}"));

        Map<String, Object> role =
                (Map<String, Object>) onlyChoice((Map<String, Object>) events.get(0)).get("delta");
        assertEquals("assistant", role.get("role"));
        StringBuilder text = new StringBuilder();
        for (Object event : events.subList(1, events.size() - 3)) {
            Map<String, Object> delta =
                    (Map<String, Object>) onlyChoice((Map<String, Object>) event).get("delta");
            text.append(delta.get("content"));
        }
        assertEquals("Hello", text.toString());
    }
}
