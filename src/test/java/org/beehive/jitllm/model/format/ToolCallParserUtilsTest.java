package org.beehive.jllm.model.format;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Optional;
import org.junit.Test;

/**
 * Unit tests for {@link ToolCallParserUtils}.
 *
 * <p>The parser is pure string-handling (no model/tokenizer), so every recognised response shape is
 * covered here: Qwen3/Llama {@code <tool_call>} tags, Llama 3.1 {@code <|python_tag|>}, raw-JSON
 * and markdown-fence fallbacks, unclosed tags, and batch (multi-call) responses. The
 * brace-in-string cases pin the fix that keeps argument objects whose string values contain {@code
 * {}/{}} characters (e.g. source code) intact.
 */
public class ToolCallParserUtilsTest {

    // ── Single-call extraction ────────────────────────────────────────────────

    @Test
    public void qwen3ToolCall_arguments() {
        Optional<ToolCallExtract> tc =
                ToolCallParserUtils.parseToolCallResponse(
                        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Chania\"}}\n</tool_call>");
        assertTrue(tc.isPresent());
        assertEquals("get_weather", tc.get().name());
        assertEquals("{\"city\": \"Chania\"}", tc.get().argumentsJson());
    }

    @Test
    public void llama31_pythonTag_parametersKey() {
        Optional<ToolCallExtract> tc =
                ToolCallParserUtils.parseToolCallResponse(
                        "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Boston\"}}");
        assertTrue(tc.isPresent());
        assertEquals("get_weather", tc.get().name());
        assertEquals("{\"city\": \"Boston\"}", tc.get().argumentsJson());
    }

    @Test
    public void functionKey_usedAsNameFallback() {
        Optional<ToolCallExtract> tc =
                ToolCallParserUtils.parseToolCallResponse(
                        "<tool_call>{\"function\": \"list_dir\", \"arguments\": {\"path\": \"/tmp\"}}</tool_call>");
        assertTrue(tc.isPresent());
        assertEquals("list_dir", tc.get().name());
    }

    @Test
    public void missingArguments_defaultsToEmptyObject() {
        Optional<ToolCallExtract> tc =
                ToolCallParserUtils.parseToolCallResponse(
                        "<tool_call>{\"name\": \"now\"}</tool_call>");
        assertTrue(tc.isPresent());
        assertEquals("now", tc.get().name());
        assertEquals("{}", tc.get().argumentsJson());
    }

    @Test
    public void unclosedToolCall_stillParsed() {
        // Model stopped (eot/eom) before emitting the closing tag.
        Optional<ToolCallExtract> tc =
                ToolCallParserUtils.parseToolCallResponse(
                        "<tool_call>{\"name\": \"ping\", \"arguments\": {\"host\": \"a\"}}");
        assertTrue(tc.isPresent());
        assertEquals("ping", tc.get().name());
        assertEquals("{\"host\": \"a\"}", tc.get().argumentsJson());
    }

    @Test
    public void plainTextResponse_isNotAToolCall() {
        assertFalse(
                ToolCallParserUtils.parseToolCallResponse("The weather in Chania is sunny.")
                        .isPresent());
    }

    // ── Brace-in-string argument objects (the core fix) ───────────────────────

    @Test
    public void argumentsWithBracesInStringValue_keptIntact() {
        String args = "{\"code\": \"public class A { void m() { return; } }\"}";
        Optional<ToolCallExtract> tc =
                ToolCallParserUtils.parseToolCallResponse(
                        "<tool_call>{\"name\": \"write_file\", \"arguments\": "
                                + args
                                + "}</tool_call>");
        assertTrue(tc.isPresent());
        assertEquals("write_file", tc.get().name());
        assertEquals(args, tc.get().argumentsJson());
    }

    @Test
    public void argumentsWithEscapedQuotesAndBraces_keptIntact() {
        String args = "{\"snippet\": \"if (s.equals(\\\"}\\\")) { x++; }\"}";
        Optional<ToolCallExtract> tc =
                ToolCallParserUtils.parseToolCallResponse(
                        "<tool_call>{\"name\": \"run\", \"arguments\": " + args + "}</tool_call>");
        assertTrue(tc.isPresent());
        assertEquals(args, tc.get().argumentsJson());
    }

    @Test
    public void argumentsWithNestedObjectsAndArrays_keptIntact() {
        String args = "{\"items\": [{\"a\": 1}, {\"b\": 2}], \"meta\": {\"n\": 3}}";
        Optional<ToolCallExtract> tc =
                ToolCallParserUtils.parseToolCallResponse(
                        "<tool_call>{\"name\": \"batch\", \"arguments\": "
                                + args
                                + "}</tool_call>");
        assertTrue(tc.isPresent());
        assertEquals(args, tc.get().argumentsJson());
    }

    // ── Fallbacks ─────────────────────────────────────────────────────────────

    @Test
    public void rawJsonFallback_noTags() {
        Optional<ToolCallExtract> tc =
                ToolCallParserUtils.parseToolCallResponse(
                        "{\"name\": \"echo\", \"arguments\": {\"msg\": \"hi\"}}");
        assertTrue(tc.isPresent());
        assertEquals("echo", tc.get().name());
    }

    @Test
    public void markdownFencedJson_fallback() {
        Optional<ToolCallExtract> tc =
                ToolCallParserUtils.parseToolCallResponse(
                        "```json\n{\"name\": \"echo\", \"arguments\": {\"msg\": \"hi\"}}\n```");
        assertTrue(tc.isPresent());
        assertEquals("echo", tc.get().name());
        assertEquals("{\"msg\": \"hi\"}", tc.get().argumentsJson());
    }

    @Test
    public void stripMarkdownFences_removesFenceLines() {
        assertEquals("body", ToolCallParserUtils.stripMarkdownFences("```\nbody\n```"));
        assertEquals("plain", ToolCallParserUtils.stripMarkdownFences("plain"));
    }

    // ── Batch (multiple tool calls) ───────────────────────────────────────────

    @Test
    public void batch_multipleToolCallBlocks() {
        List<ToolCallExtract> calls =
                ToolCallParserUtils.parseAllToolCalls(
                        "<tool_call>{\"name\": \"a\", \"arguments\": {\"x\": 1}}</tool_call>"
                                + "<tool_call>{\"name\": \"b\", \"arguments\": {\"y\": 2}}</tool_call>");
        assertEquals(2, calls.size());
        assertEquals("a", calls.get(0).name());
        assertEquals("{\"x\": 1}", calls.get(0).argumentsJson());
        assertEquals("b", calls.get(1).name());
        assertEquals("{\"y\": 2}", calls.get(1).argumentsJson());
    }

    @Test
    public void batch_bracesInStringDoNotBleedAcrossCalls() {
        List<ToolCallExtract> calls =
                ToolCallParserUtils.parseAllToolCalls(
                        "<tool_call>{\"name\": \"write\", \"arguments\": {\"code\": \"a { b }\"}}</tool_call>"
                                + "<tool_call>{\"name\": \"log\", \"arguments\": {\"msg\": \"ok\"}}</tool_call>");
        assertEquals(2, calls.size());
        assertEquals("{\"code\": \"a { b }\"}", calls.get(0).argumentsJson());
        assertEquals("log", calls.get(1).name());
    }

    @Test
    public void batch_pythonTagIsSingleCall() {
        List<ToolCallExtract> calls =
                ToolCallParserUtils.parseAllToolCalls(
                        "<|python_tag|>{\"name\": \"a\", \"parameters\": {\"x\": 1}}");
        assertEquals(1, calls.size());
        assertEquals("a", calls.get(0).name());
    }

    @Test
    public void batch_noToolCalls_returnsEmpty() {
        assertTrue(ToolCallParserUtils.parseAllToolCalls("just a plain answer").isEmpty());
    }
}
