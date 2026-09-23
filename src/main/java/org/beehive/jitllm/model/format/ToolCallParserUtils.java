package org.beehive.jllm.model.format;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Pure-string tool-call extraction for Llama and Qwen3 response formats.
 *
 * <p>All methods are stateless and do not require any model or tokenizer instance, making them
 * directly unit-testable.
 */
public final class ToolCallParserUtils {

    private ToolCallParserUtils() {}

    /**
     * Extracts a single tool call from a model response text.
     *
     * <p>Recognised formats (in priority order): 1. {@code <|python_tag|>{…}} — LLaMA 3.1 native 2.
     * {@code <tool_call>…</tool_call>} — LLaMA 3.2 and Qwen3 (closed or unclosed) 3. Raw JSON
     * object optionally inside markdown code fences — fallback
     *
     * <p>Both {@code "parameters"} and {@code "arguments"} are tried as the argument key, covering
     * LLaMA 3.1/3.2 and Qwen3 variants transparently.
     */
    public static Optional<ToolCallExtract> parseToolCallResponse(String responseText) {
        // 1. Native LLaMA 3.1 format: <|python_tag|>{.}
        int idx = responseText.indexOf("<|python_tag|>");
        if (idx != -1) {
            String json = responseText.substring(idx + "<|python_tag|>".length()).strip();
            return parseToolCallJson(json);
        }

        // 2. LLaMA 3.2 format: <tool_call>.</tool_call>
        int tcStart = responseText.indexOf("<tool_call>");
        int tcEnd = responseText.lastIndexOf("</tool_call>");
        if (tcStart != -1 && tcEnd != -1 && tcEnd > tcStart) {
            String json = responseText.substring(tcStart + "<tool_call>".length(), tcEnd).strip();
            return parseToolCallJson(json);
        }
        // 2b. Unclosed <tool_call> — model stopped (eot_id / eom_id) before writing the closing tag
        if (tcStart != -1 && tcEnd == -1) {
            String json = responseText.substring(tcStart + "<tool_call>".length()).strip();
            return parseToolCallJson(json);
        }

        // 3. Fallback: raw JSON, possibly inside markdown code fences
        String stripped = stripMarkdownFences(responseText.strip());
        if (stripped.startsWith("{")) {
            return parseToolCallJson(stripped);
        }

        return Optional.empty();
    }

    /**
     * Parses a tool call JSON object extracted from a {@code <tool_call>} block or raw JSON.
     * Accepts {@code {"name":…,"parameters":{…}}}, {@code {"function":…,"parameters":{…}}}, and
     * {@code {"name":…,"arguments":{…}}} — covering both LLaMA and Qwen3 variants.
     */
    private static Optional<ToolCallExtract> parseToolCallJson(String json) {
        String name = extractStringValue(json, "name");
        if (name == null) {
            name = extractStringValue(json, "function");
        }
        if (name == null) return Optional.empty();

        String argsJson = extractNestedObject(json, "parameters");
        if (argsJson == null) argsJson = extractNestedObject(json, "arguments");
        if (argsJson == null) argsJson = "{}";

        return Optional.of(new ToolCallExtract(name, argsJson));
    }

    // Batch extraction

    /**
     * Extracts ALL tool calls from a response that may contain multiple {@code
     * <tool_call>…</tool_call>} blocks (Llama 3.2 and Qwen3 batch calls).
     *
     * <p>Falls back to the raw-JSON single-call path if no tags are found. Returns an empty list
     * when the response contains no tool calls.
     */
    public static List<ToolCallExtract> parseAllToolCalls(String responseText) {
        List<ToolCallExtract> calls = new ArrayList<>();

        // <|python_tag|> (Llama 3.1) — single call by definition
        int pythonIdx = responseText.indexOf("<|python_tag|>");
        if (pythonIdx != -1) {
            parseToolCallJson(responseText.substring(pythonIdx + "<|python_tag|>".length()).strip())
                    .ifPresent(calls::add);
            return calls;
        }

        // Scan for all <tool_call>…</tool_call> blocks
        int searchFrom = 0;
        while (true) {
            int start = responseText.indexOf("<tool_call>", searchFrom);
            if (start == -1) break;
            int end = responseText.indexOf("</tool_call>", start);
            String json;
            if (end != -1) {
                json = responseText.substring(start + "<tool_call>".length(), end).strip();
                searchFrom = end + "</tool_call>".length();
            } else {
                // Unclosed tag — model stopped before writing the closing tag
                json = responseText.substring(start + "<tool_call>".length()).strip();
                searchFrom = responseText.length();
            }
            parseToolCallJson(json).ifPresent(calls::add);
            if (end == -1) break;
        }

        // Raw JSON fallback (no tags at all)
        if (calls.isEmpty()) {
            String stripped = stripMarkdownFences(responseText.strip());
            if (stripped.startsWith("{")) {
                parseToolCallJson(stripped).ifPresent(calls::add);
            }
        }

        return calls;
    }

    // Shared helpers

    /** Strips surrounding markdown code fences (```…```) if present. */
    public static String stripMarkdownFences(String text) {
        if (!text.startsWith("```")) return text;
        int firstNewline = text.indexOf('\n');
        if (firstNewline == -1) return text;
        String body = text.substring(firstNewline + 1);
        if (body.endsWith("```")) body = body.substring(0, body.length() - 3).stripTrailing();
        return body.strip();
    }

    /**
     * Extracts the string value for {@code "key": "<value>"} from a JSON object. Tolerates
     * whitespace around {@code:} and correctly skips escaped quotes ({@code \"}) inside the value,
     * so multi-line code strings with embedded {@code "} are returned intact.
     */
    private static String extractStringValue(String json, String key) {
        String marker = "\"" + key + "\"";
        int markerIdx = json.indexOf(marker);
        if (markerIdx == -1) return null;
        int colonIdx = json.indexOf(':', markerIdx + marker.length());
        if (colonIdx == -1) return null;
        int quoteStart = json.indexOf('"', colonIdx + 1);
        if (quoteStart == -1) return null;
        // Scan for the closing quote, honouring backslash escapes
        int i = quoteStart + 1;
        while (i < json.length()) {
            char c = json.charAt(i);
            if (c == '\\') {
                i += 2; // skip escape sequence (e.g. \", \\, \n)
            } else if (c == '"') {
                break;
            } else {
                i++;
            }
        }
        if (i >= json.length()) return null;
        return json.substring(quoteStart + 1, i);
    }

    /**
     * Extracts the JSON object value for {@code "key": {…}} using brace-counting. Handles nested
     * objects and tolerates whitespace around {@code:}.
     *
     * <p>Brace counting is <em>string-aware</em>: {@code {} and } characters appearing inside JSON
     * string literals (e.g. a {@code "code"} argument containing Java source) do not affect the
     * depth counter, and {@code \"} escapes inside strings are skipped. This keeps argument objects
     * whose string values contain braces intact.
     */
    private static String extractNestedObject(String json, String key) {
        String marker = "\"" + key + "\"";
        int markerIdx = json.indexOf(marker);
        if (markerIdx == -1) return null;
        int colonIdx = json.indexOf(':', markerIdx + marker.length());
        if (colonIdx == -1) return null;
        int braceStart = json.indexOf('{', colonIdx + 1);
        if (braceStart == -1) return null;
        int depth = 0;
        boolean inString = false;
        for (int i = braceStart; i < json.length(); i++) {
            char c = json.charAt(i);
            if (inString) {
                if (c == '\\') {
                    i++; // skip the escaped character (e.g. \", \\, \n)
                } else if (c == '"') {
                    inString = false;
                }
            } else if (c == '"') {
                inString = true;
            } else if (c == '{') {
                depth++;
            } else if (c == '}') {
                if (--depth == 0) return json.substring(braceStart, i + 1);
            }
        }
        return null; // unbalanced
    }
}
