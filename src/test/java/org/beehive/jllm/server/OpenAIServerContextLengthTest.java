package org.beehive.jllm.server;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.Test;

/**
 * The context length the server advertises, and the rule that picks it — without a model in sight.
 * What a client reads off {@code /v1/models} is the only thing it can size a prompt against, so
 * both halves are worth pinning: the resolution rule, and the JSON shape.
 */
public class OpenAIServerContextLengthTest {

    // ---- resolveContextLength -------------------------------------------------------------

    @Test
    public void noRequestMeansTheModelsOwnContext() {
        assertEquals(131072, OpenAIServer.resolveContextLength(0, 131072));
    }

    @Test
    public void aNegativeRequestIsTreatedAsNoRequest() {
        assertEquals(8192, OpenAIServer.resolveContextLength(-1, 8192));
    }

    @Test
    public void aSmallerRequestIsHonoured() {
        assertEquals(4096, OpenAIServer.resolveContextLength(4096, 131072));
    }

    /** The loaders clamp rather than fail, so the advertised value has to clamp the same way. */
    @Test
    public void aRequestBeyondTheModelIsClampedToTheModel() {
        assertEquals(8192, OpenAIServer.resolveContextLength(1_000_000, 8192));
    }

    @Test
    public void aRequestEqualToTheModelIsUnchanged() {
        assertEquals(8192, OpenAIServer.resolveContextLength(8192, 8192));
    }

    // ---- modelsPayload --------------------------------------------------------------------

    @SuppressWarnings("unchecked")
    private static Map<String, Object> onlyEntry(Map<String, Object> payload) {
        List<Object> data = (List<Object>) payload.get("data");
        assertEquals("the server serves exactly one model", 1, data.size());
        return (Map<String, Object>) data.get(0);
    }

    @Test
    public void thePayloadIsAnOpenAiModelList() {
        Map<String, Object> payload =
                OpenAIServer.modelsPayload("Llama-3.2-1B-Instruct-Q8_0", 8192);

        assertEquals("list", payload.get("object"));
        Map<String, Object> entry = onlyEntry(payload);
        assertEquals("Llama-3.2-1B-Instruct-Q8_0", entry.get("id"));
        assertEquals("model", entry.get("object"));
        assertEquals(0, entry.get("created"));
        assertEquals("jllm", entry.get("owned_by"));
    }

    @Test
    public void aKnownContextLengthIsAdvertised() {
        Map<String, Object> entry =
                onlyEntry(OpenAIServer.modelsPayload("Llama-3.2-1B-Instruct-Q8_0", 131072));

        assertEquals(131072, entry.get("context_length"));
    }

    /**
     * Zero is not a context length any client should believe, so it is left out entirely: a reader
     * that finds no field falls back to its own default, one that finds 0 does not.
     */
    @Test
    public void anUnknownContextLengthIsOmittedRatherThanPublishedAsZero() {
        Map<String, Object> entry =
                onlyEntry(
                        OpenAIServer.modelsPayload(
                                "Llama-3.2-1B-Instruct-Q8_0", OpenAIServer.UNKNOWN_CONTEXT_LENGTH));

        assertFalse(
                "context_length must be absent when unknown", entry.containsKey("context_length"));
        assertTrue("the rest of the entry still stands", entry.containsKey("id"));
    }

    @Test
    public void theEntryKeepsTheOpenAiFieldOrder() {
        Map<String, Object> entry =
                onlyEntry(OpenAIServer.modelsPayload("Llama-3.2-1B-Instruct-Q8_0", 8192));

        assertEquals(
                List.of("id", "object", "created", "owned_by", "context_length"),
                List.copyOf(entry.keySet()));
    }
}
