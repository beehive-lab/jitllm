package org.beehive.jitllm.model.format;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * LFM2's tool calls: a Python-style list of keyword calls between {@code <|tool_call_start|>} and
 * {@code <|tool_call_end|>}, for example {@code [getWeather(city='Paris'), get_current_time()]}.
 *
 * <p>{@link #render} writes calls as LFM2.5's template writes {@code tool_calls}: a string argument
 * single-quoted with {@code \\}, {@code '}, newline and carriage return escaped, an object or list
 * through {@code tojson}, anything else as Python's {@code str} spells it ({@code True}, {@code
 * None}). {@link #parse} reads that syntax back, and also double-quoted strings and JSON's {@code
 * true}, {@code false} and {@code null}, into a JSON arguments object.
 */
final class Lfm2ToolCalls {

    static final String START = "<|tool_call_start|>";
    static final String END = "<|tool_call_end|>";

    private final String s;
    private int i;

    private Lfm2ToolCalls(String s) {
        this.s = s;
    }

    /** {@code [name(key=value, ...), ...]}, the text between the two markers. */
    static String render(List<ToolCallExtract> calls) {
        StringBuilder out = new StringBuilder("[");
        for (int c = 0; c < calls.size(); c++) {
            if (c > 0) {
                out.append(", ");
            }
            ToolCallExtract call = calls.get(c);
            out.append(call.name()).append('(');
            boolean first = true;
            for (Map.Entry<String, Object> argument : arguments(call).entrySet()) {
                if (!first) {
                    out.append(", ");
                }
                first = false;
                out.append(argument.getKey()).append('=').append(value(argument.getValue()));
            }
            out.append(')');
        }
        return out.append(']').toString();
    }

    /** The call's arguments object; blank, {@code {}} and {@code null} are none. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> arguments(ToolCallExtract call) {
        String json = call.argumentsJson();
        if (json == null || json.isBlank()) {
            return Map.of();
        }
        Object parsed = ToolJson.parse(json);
        if (parsed == ToolJson.NULL) {
            return Map.of();
        }
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IllegalArgumentException(
                    "LFM2 writes a tool call's arguments as keywords, so they must be a JSON object: "
                            + json);
        }
        return (Map<String, Object>) map;
    }

    private static String value(Object value) {
        if (value instanceof String text) {
            return "'"
                    + text.replace("\\", "\\\\")
                            .replace("'", "\\'")
                            .replace("\n", "\\n")
                            .replace("\r", "\\r")
                    + "'";
        }
        if (value instanceof Map<?, ?> || value instanceof List<?>) {
            return ToolJson.dumps(value);
        }
        if (value instanceof Boolean flag) {
            return flag ? "True" : "False";
        }
        if (value == ToolJson.NULL) {
            return "None";
        }
        return String.valueOf(value);
    }

    /**
     * Every call in every {@code <|tool_call_start|>…<|tool_call_end|>} block of {@code text}, in
     * order; empty when there is none.
     *
     * @throws IllegalArgumentException when a block does not parse
     */
    static List<ToolCallExtract> parse(String text) {
        List<ToolCallExtract> calls = new ArrayList<>();
        int from = 0;
        while (true) {
            int start = text.indexOf(START, from);
            if (start < 0) {
                return calls;
            }
            int end = text.indexOf(END, start + START.length());
            if (end < 0) {
                throw new IllegalArgumentException("unclosed " + START);
            }
            new Lfm2ToolCalls(text.substring(start + START.length(), end)).calls(calls);
            from = end + END.length();
        }
    }

    private void calls(List<ToolCallExtract> calls) {
        expect('[');
        if (!peek(']')) {
            do {
                calls.add(call());
            } while (accept(','));
        }
        expect(']');
        skipSpace();
        if (i != s.length()) {
            throw error("text after the list");
        }
    }

    private ToolCallExtract call() {
        String name = name();
        expect('(');
        Map<String, Object> arguments = new LinkedHashMap<>();
        if (!peek(')')) {
            do {
                String key = name();
                expect('=');
                arguments.put(key, literal());
            } while (accept(','));
        }
        expect(')');
        return new ToolCallExtract(name, ToolJson.dumps(arguments));
    }

    private String name() {
        skipSpace();
        int start = i;
        while (i < s.length()) {
            char c = s.charAt(i);
            if (!(Character.isLetterOrDigit(c) || c == '_' || c == '.' || c == '-')) {
                break;
            }
            i++;
        }
        if (i == start) {
            throw error("a name");
        }
        return s.substring(start, i);
    }

    private Object literal() {
        skipSpace();
        if (i >= s.length()) {
            throw error("a value");
        }
        char c = s.charAt(i);
        if (c == '\'' || c == '"') {
            return string(c);
        }
        if (c == '[') {
            i++;
            List<Object> items = new ArrayList<>();
            if (!peek(']')) {
                do {
                    items.add(literal());
                } while (accept(','));
            }
            expect(']');
            return items;
        }
        if (c == '{') {
            i++;
            Map<String, Object> entries = new LinkedHashMap<>();
            if (!peek('}')) {
                do {
                    skipSpace();
                    char quote = i < s.length() ? s.charAt(i) : 0;
                    if (quote != '\'' && quote != '"') {
                        throw error("a quoted key");
                    }
                    String key = string(quote);
                    expect(':');
                    entries.put(key, literal());
                } while (accept(','));
            }
            expect('}');
            return entries;
        }
        int start = i;
        while (i < s.length() && ",)]}".indexOf(s.charAt(i)) < 0) {
            i++;
        }
        String word = s.substring(start, i).strip();
        return switch (word) {
            case "True", "true" -> Boolean.TRUE;
            case "False", "false" -> Boolean.FALSE;
            case "None", "null" -> ToolJson.NULL;
            default -> {
                if (!ToolJson.isNumber(word)) {
                    throw error("a literal, not '" + word + "'");
                }
                yield new ToolJson.Number(word);
            }
        };
    }

    private String string(char quote) {
        i++;
        StringBuilder out = new StringBuilder();
        while (i < s.length()) {
            char c = s.charAt(i++);
            if (c == quote) {
                return out.toString();
            }
            if (c != '\\') {
                out.append(c);
                continue;
            }
            if (i >= s.length()) {
                break;
            }
            char e = s.charAt(i++);
            switch (e) {
                case 'n' -> out.append('\n');
                case 'r' -> out.append('\r');
                case 't' -> out.append('\t');
                case 'u' -> {
                    if (i + 4 > s.length()) {
                        throw error("four hex digits");
                    }
                    out.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                    i += 4;
                }
                default -> out.append(e);
            }
        }
        throw error("a closing " + quote);
    }

    private void skipSpace() {
        while (i < s.length() && Character.isWhitespace(s.charAt(i))) {
            i++;
        }
    }

    private boolean peek(char c) {
        skipSpace();
        return i < s.length() && s.charAt(i) == c;
    }

    private boolean accept(char c) {
        if (peek(c)) {
            i++;
            return true;
        }
        return false;
    }

    private void expect(char c) {
        if (!accept(c)) {
            throw error("'" + c + "'");
        }
    }

    private IllegalArgumentException error(String wanted) {
        return new IllegalArgumentException(
                "LFM2 tool call: expected " + wanted + " at " + i + " in " + s);
    }
}
