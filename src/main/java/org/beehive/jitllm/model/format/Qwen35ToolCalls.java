package org.beehive.jllm.model.format;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * The {@code qwen35} tool-call wire format, in both directions.
 *
 * <p>Qwen 3.5 abandoned the JSON body Qwen 3 put inside {@code <tool_call>} and replaced it with
 * nested pseudo-XML, one element per argument:
 *
 * <pre>
 *   &lt;tool_call&gt;
 *   &lt;function=get_weather&gt;
 *   &lt;parameter=location&gt;
 *   Boston
 *   &lt;/parameter&gt;
 *   &lt;parameter=units&gt;
 *   celsius
 *   &lt;/parameter&gt;
 *   &lt;/function&gt;
 *   &lt;/tool_call&gt;
 * </pre>
 *
 * <p>The engine's own currency is a name and a JSON object ({@link ToolCallExtract}), so this class
 * is the translation at the boundary and nothing above it changes.
 *
 * <h2>Typing, which the format erases</h2>
 *
 * <p>A parameter's body is bare text. The chat template writes a string argument verbatim and every
 * other kind through {@code tojson}, so {@code Boston} and {@code 42} and {@code {"a":1}} are all
 * unquoted in the same position, and only their shape says which is which. Reading them back,
 * anything that is a well-formed JSON scalar, object or array is taken as that; everything else is
 * a string and is quoted. That is the inverse of what the template did, and it is a heuristic — a
 * tool whose string argument is literally {@code "42"} round-trips as the number 42. The
 * alternative, quoting everything, breaks every numeric and boolean parameter, which is the far
 * commoner case.
 *
 * <p>Multi-line values survive: the delimiters are whole lines, and only the newline immediately
 * inside each tag is stripped.
 */
public final class Qwen35ToolCalls {

    private static final String CALL_OPEN = "<tool_call>";
    private static final String CALL_CLOSE = "</tool_call>";
    private static final String FUNCTION_OPEN = "<function=";
    private static final String FUNCTION_CLOSE = "</function>";
    private static final String PARAMETER_OPEN = "<parameter=";
    private static final String PARAMETER_CLOSE = "</parameter>";

    private Qwen35ToolCalls() {}

    // ---- model output → ToolCallExtract --------------------------------------

    /**
     * Every tool call in a response, in the order the model wrote them.
     *
     * <p>An unclosed trailing block is parsed too: a model that stops on its token budget mid-call
     * has still said which function it wants, and dropping that silently is worse than acting on a
     * truncated argument list a caller can validate.
     */
    public static List<ToolCallExtract> parseAll(String responseText) {
        List<ToolCallExtract> calls = new ArrayList<>();
        if (responseText == null) {
            return calls;
        }
        int from = 0;
        while (true) {
            int start = responseText.indexOf(CALL_OPEN, from);
            if (start == -1) {
                break;
            }
            int bodyStart = start + CALL_OPEN.length();
            int end = responseText.indexOf(CALL_CLOSE, bodyStart);
            String body =
                    end == -1
                            ? responseText.substring(bodyStart)
                            : responseText.substring(bodyStart, end);
            parseOne(body).ifPresent(calls::add);
            if (end == -1) {
                break;
            }
            from = end + CALL_CLOSE.length();
        }
        return calls;
    }

    /** The first tool call in a response, or empty. */
    public static Optional<ToolCallExtract> parseFirst(String responseText) {
        List<ToolCallExtract> all = parseAll(responseText);
        return all.isEmpty() ? Optional.empty() : Optional.of(all.get(0));
    }

    /** One {@code <function=…>} block's contents, without the surrounding call tags. */
    private static Optional<ToolCallExtract> parseOne(String body) {
        int open = body.indexOf(FUNCTION_OPEN);
        if (open == -1) {
            return Optional.empty();
        }
        int nameEnd = body.indexOf('>', open + FUNCTION_OPEN.length());
        if (nameEnd == -1) {
            return Optional.empty();
        }
        String name = body.substring(open + FUNCTION_OPEN.length(), nameEnd).strip();
        if (name.isEmpty()) {
            return Optional.empty();
        }

        int close = body.indexOf(FUNCTION_CLOSE, nameEnd);
        String params =
                close == -1 ? body.substring(nameEnd + 1) : body.substring(nameEnd + 1, close);

        StringBuilder json = new StringBuilder("{");
        int from = 0;
        boolean first = true;
        while (true) {
            int pOpen = params.indexOf(PARAMETER_OPEN, from);
            if (pOpen == -1) {
                break;
            }
            int pNameEnd = params.indexOf('>', pOpen + PARAMETER_OPEN.length());
            if (pNameEnd == -1) {
                break;
            }
            String key = params.substring(pOpen + PARAMETER_OPEN.length(), pNameEnd).strip();
            int pClose = params.indexOf(PARAMETER_CLOSE, pNameEnd);
            String raw =
                    pClose == -1
                            ? params.substring(pNameEnd + 1)
                            : params.substring(pNameEnd + 1, pClose);
            if (!first) {
                json.append(',');
            }
            first = false;
            json.append(quote(key)).append(':').append(asJsonValue(trimTagNewlines(raw)));
            if (pClose == -1) {
                break;
            }
            from = pClose + PARAMETER_CLOSE.length();
        }
        json.append('}');
        return Optional.of(new ToolCallExtract(name, json.toString()));
    }

    /**
     * Drops the newline the format puts immediately after the opening tag and before the closing
     * one, and nothing else — a value's own blank lines and indentation are its own.
     */
    private static String trimTagNewlines(String raw) {
        String value = raw;
        if (value.startsWith("\r\n")) {
            value = value.substring(2);
        } else if (value.startsWith("\n")) {
            value = value.substring(1);
        }
        if (value.endsWith("\r\n")) {
            value = value.substring(0, value.length() - 2);
        } else if (value.endsWith("\n")) {
            value = value.substring(0, value.length() - 1);
        }
        return value;
    }

    // ---- ToolCallExtract → model input ---------------------------------------

    /**
     * Renders a call back into the wire format, for replaying an assistant turn in history.
     *
     * @return the {@code <function=…>…</function>} block, without the {@code <tool_call>} tags
     */
    public static String renderFunctionBlock(ToolCallExtract call) {
        StringBuilder out = new StringBuilder();
        out.append(FUNCTION_OPEN).append(call.name()).append(">\n");
        for (String[] entry : topLevelEntries(call.argumentsJson())) {
            out.append(PARAMETER_OPEN)
                    .append(entry[0])
                    .append(">\n")
                    .append(entry[1])
                    .append("\n")
                    .append(PARAMETER_CLOSE)
                    .append('\n');
        }
        out.append(FUNCTION_CLOSE);
        return out.toString();
    }

    /**
     * Splits a JSON object into its top-level entries, as {@code {key, renderedValue}} pairs.
     *
     * <p>String values are unescaped to their text — that is how the template writes them — and
     * every other value is emitted as the JSON it already is. Written by hand rather than with a
     * parser because the whole job is two levels deep and the project carries no JSON dependency.
     */
    private static List<String[]> topLevelEntries(String json) {
        List<String[]> entries = new ArrayList<>();
        if (json == null) {
            return entries;
        }
        String s = json.strip();
        if (!s.startsWith("{") || !s.endsWith("}")) {
            return entries;
        }
        int i = 1;
        int end = s.length() - 1;
        while (i < end) {
            while (i < end && (Character.isWhitespace(s.charAt(i)) || s.charAt(i) == ',')) {
                i++;
            }
            if (i >= end || s.charAt(i) != '"') {
                break;
            }
            int keyEnd = endOfString(s, i);
            if (keyEnd == -1) {
                break;
            }
            String key = unescape(s.substring(i + 1, keyEnd));
            i = keyEnd + 1;
            while (i < end && Character.isWhitespace(s.charAt(i))) {
                i++;
            }
            if (i >= end || s.charAt(i) != ':') {
                break;
            }
            i++;
            while (i < end && Character.isWhitespace(s.charAt(i))) {
                i++;
            }
            int valueEnd = endOfValue(s, i, end);
            if (valueEnd == -1) {
                break;
            }
            String rawValue = s.substring(i, valueEnd).strip();
            String rendered =
                    rawValue.startsWith("\"")
                            ? unescape(rawValue.substring(1, rawValue.length() - 1))
                            : rawValue;
            entries.add(new String[] {key, rendered});
            i = valueEnd;
        }
        return entries;
    }

    /** Index of the closing quote of the string starting at {@code start}, or -1. */
    private static int endOfString(String s, int start) {
        for (int i = start + 1; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '\\') {
                i++;
            } else if (c == '"') {
                return i;
            }
        }
        return -1;
    }

    /** Index just past the value starting at {@code start}, or -1 when it is unterminated. */
    private static int endOfValue(String s, int start, int limit) {
        char c = s.charAt(start);
        if (c == '"') {
            int close = endOfString(s, start);
            return close == -1 ? -1 : close + 1;
        }
        if (c == '{' || c == '[') {
            char open = c;
            char shut = c == '{' ? '}' : ']';
            int depth = 0;
            boolean inString = false;
            for (int i = start; i < s.length(); i++) {
                char d = s.charAt(i);
                if (inString) {
                    if (d == '\\') {
                        i++;
                    } else if (d == '"') {
                        inString = false;
                    }
                } else if (d == '"') {
                    inString = true;
                } else if (d == open) {
                    depth++;
                } else if (d == shut && --depth == 0) {
                    return i + 1;
                }
            }
            return -1;
        }
        int i = start;
        while (i < limit && s.charAt(i) != ',' && !Character.isWhitespace(s.charAt(i))) {
            i++;
        }
        return i;
    }

    // ---- JSON scalars --------------------------------------------------------

    /** Whether a parameter body is already well-formed JSON, and should be kept as it stands. */
    private static String asJsonValue(String value) {
        String v = value.strip();
        if (v.equals("true") || v.equals("false") || v.equals("null")) {
            return v;
        }
        if (isJsonNumber(v)) {
            return v;
        }
        if ((v.startsWith("{") && v.endsWith("}")) || (v.startsWith("[") && v.endsWith("]"))) {
            return v;
        }
        return quote(value);
    }

    private static boolean isJsonNumber(String v) {
        if (v.isEmpty()) {
            return false;
        }
        int i = v.charAt(0) == '-' ? 1 : 0;
        if (i >= v.length()) {
            return false;
        }
        boolean digits = false;
        boolean dot = false;
        boolean exponent = false;
        for (; i < v.length(); i++) {
            char c = v.charAt(i);
            if (c >= '0' && c <= '9') {
                digits = true;
            } else if (c == '.' && !dot && !exponent) {
                dot = true;
            } else if ((c == 'e' || c == 'E') && digits && !exponent) {
                exponent = true;
                if (i + 1 < v.length() && (v.charAt(i + 1) == '+' || v.charAt(i + 1) == '-')) {
                    i++;
                }
            } else {
                return false;
            }
        }
        return digits;
    }

    /** A JSON string literal, quotes included. */
    static String quote(String text) {
        StringBuilder out = new StringBuilder(text.length() + 2).append('"');
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            switch (c) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                case '\b' -> out.append("\\b");
                case '\f' -> out.append("\\f");
                default -> {
                    if (c < 0x20) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
                }
            }
        }
        return out.append('"').toString();
    }

    /** The text a JSON string literal's body stands for. */
    static String unescape(String body) {
        StringBuilder out = new StringBuilder(body.length());
        for (int i = 0; i < body.length(); i++) {
            char c = body.charAt(i);
            if (c != '\\' || i + 1 >= body.length()) {
                out.append(c);
                continue;
            }
            char next = body.charAt(++i);
            switch (next) {
                case 'n' -> out.append('\n');
                case 'r' -> out.append('\r');
                case 't' -> out.append('\t');
                case 'b' -> out.append('\b');
                case 'f' -> out.append('\f');
                case '"' -> out.append('"');
                case '\\' -> out.append('\\');
                case '/' -> out.append('/');
                case 'u' -> {
                    if (i + 4 < body.length()) {
                        out.append((char) Integer.parseInt(body.substring(i + 1, i + 5), 16));
                        i += 4;
                    }
                }
                default -> out.append(next);
            }
        }
        return out.toString();
    }
}
