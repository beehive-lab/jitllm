package org.beehive.jitllm.server;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Map;
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
}
