package org.beehive.jitllm.model.format;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.List;
import org.junit.Test;

/** LFM2's Python-style tool calls, written as its template writes them and read back. */
public class Lfm2ToolCallsTest {

    /** As LFM2.5's {@code render_tool_calls} writes the scenario calls. */
    @Test
    public void rendersAsTheTemplateDoes() {
        assertEquals(
                "[getWeather(city='Paris'), get_current_time()]",
                Lfm2ToolCalls.render(
                        List.of(
                                new ToolCallExtract("getWeather", "{\"city\":\"Paris\"}"),
                                new ToolCallExtract("get_current_time", "{}"))));
    }

    /**
     * {@code format_arg_value}: strings single-quoted with backslash, quote, newline and carriage
     * return escaped; objects and lists through {@code tojson}; the rest as Python's {@code str}.
     */
    @Test
    public void rendersEachKindOfValue() {
        String json =
                "{\"s\":\"it's a\\\\b\\nc\",\"n\":18,\"f\":1.5,\"t\":true,\"z\":null,"
                        + "\"l\":[1,\"x\"],\"o\":{\"k\":\"v\"}}";
        assertEquals(
                "[f(s='it\\'s a\\\\b\\nc', n=18, f=1.5, t=True, z=None, l=[1, \"x\"],"
                        + " o={\"k\": \"v\"})]",
                Lfm2ToolCalls.render(List.of(new ToolCallExtract("f", json))));
    }

    @Test
    public void argumentsThatAreNotAnObjectAreRefused() {
        assertThrows(
                IllegalArgumentException.class,
                () -> Lfm2ToolCalls.render(List.of(new ToolCallExtract("f", "[1]"))));
    }

    /** What the model writes: reasoning, then the block, then text. */
    @Test
    public void readsTheCallsInAResponse() {
        List<ToolCallExtract> calls =
                Lfm2ToolCalls.parse(
                        "<think>Need the weather.</think><|tool_call_start|>"
                                + "[getWeather(city=\"Paris\"), get_current_time()]"
                                + "<|tool_call_end|>Checking.");
        assertEquals(2, calls.size());
        assertEquals("getWeather", calls.get(0).name());
        assertEquals("{\"city\": \"Paris\"}", calls.get(0).argumentsJson());
        assertEquals("get_current_time", calls.get(1).name());
        assertEquals("{}", calls.get(1).argumentsJson());
    }

    /** Whatever {@link Lfm2ToolCalls#render} writes reads back to the same arguments. */
    @Test
    public void rendersAndReadsBack() {
        String json =
                "{\"s\": \"it's a\\\\b\\nc\", \"n\": 18, \"f\": -1.5e3, \"t\": true, \"z\": null,"
                        + " \"l\": [1, \"x\", false], \"o\": {\"k\": \"v\"}}";
        String text =
                Lfm2ToolCalls.START
                        + Lfm2ToolCalls.render(List.of(new ToolCallExtract("search.web", json)))
                        + Lfm2ToolCalls.END;
        List<ToolCallExtract> calls = Lfm2ToolCalls.parse(text);
        assertEquals(1, calls.size());
        assertEquals("search.web", calls.get(0).name());
        assertEquals(json, calls.get(0).argumentsJson());
    }

    @Test
    public void plainTextHasNoCalls() {
        assertTrue(Lfm2ToolCalls.parse("Paris is the capital of France.").isEmpty());
    }

    /** Tool-shaped text that does not parse is reported, so the encoder can treat it as text. */
    @Test
    public void malformedBlocksAreRefused() {
        for (String block :
                new String[] {
                    "[getWeather(city=Paris)]",
                    "[getWeather('Paris')]",
                    "[getWeather(city='Paris']",
                    "getWeather(city='Paris')",
                    "[getWeather(city='Paris')] trailing"
                }) {
            assertThrows(
                    block,
                    IllegalArgumentException.class,
                    () -> Lfm2ToolCalls.parse(Lfm2ToolCalls.START + block + Lfm2ToolCalls.END));
        }
        assertThrows(
                IllegalArgumentException.class,
                () -> Lfm2ToolCalls.parse(Lfm2ToolCalls.START + "[f()]"));
    }
}
