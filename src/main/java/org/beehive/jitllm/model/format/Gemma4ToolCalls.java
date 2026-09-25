package org.beehive.jitllm.model.format;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * The Gemma 4 tool-calling wire format, in both directions.
 *
 * <p>Everything here follows the chat template embedded in the Gemma 4 GGUF files ({@code
 * tokenizer.chat_template}); the macros it mirrors are named on each method. Gemma 4 does not put
 * JSON in the prompt. Tool declarations, calls and results are written in a compact dictionary
 * syntax whose strings are delimited by the {@code <|"|>} token rather than by quotes, and whose
 * keys are bare:
 *
 * <pre>
 *   &lt;|tool&gt;declaration:getWeather{description:&lt;|"|&gt;…&lt;|"|&gt;,parameters:{…}}&lt;tool|&gt;
 *   &lt;|tool_call&gt;call:getWeather{city:&lt;|"|&gt;Paris&lt;|"|&gt;}&lt;tool_call|&gt;
 *   &lt;|tool_response&gt;response:getWeather{value:&lt;|"|&gt;…&lt;|"|&gt;}&lt;tool_response|&gt;
 * </pre>
 *
 * <p>The engine's own currency is a name and a JSON object ({@link ToolCallExtract}), so this class
 * is the translation at the boundary and nothing above it changes.
 *
 * <p>Rendering produces {@link Piece}s rather than a string: the markers are single tokens in the
 * vocabulary and the model emits them as such, so the format encodes them as their token ids and
 * never lets text supplied by a caller (a description, an argument, a tool result) turn into one.
 */
public final class Gemma4ToolCalls {

    /** {@code <|tool>}: opens one tool declaration in the system turn. */
    public static final String TOOL_OPEN = "<|tool>";

    /** {@code <tool|>}: closes one tool declaration. */
    public static final String TOOL_CLOSE = "<tool|>";

    /** {@code <|tool_call>}: opens one call the model makes. */
    public static final String CALL_OPEN = "<|tool_call>";

    /** {@code <tool_call|>}: closes one call. */
    public static final String CALL_CLOSE = "<tool_call|>";

    /** {@code <|tool_response>}: opens one tool result; the model's end-of-call signal. */
    public static final String RESPONSE_OPEN = "<|tool_response>";

    /** {@code <tool_response|>}: closes one tool result. */
    public static final String RESPONSE_CLOSE = "<tool_response|>";

    /** {@code <|"|>}: the string delimiter of the dictionary syntax. */
    public static final String QUOTE = "<|\"|>";

    /** Every marker this format writes as a single token. */
    public static final List<String> MARKERS =
            List.of(
                    TOOL_OPEN,
                    TOOL_CLOSE,
                    CALL_OPEN,
                    CALL_CLOSE,
                    RESPONSE_OPEN,
                    RESPONSE_CLOSE,
                    QUOTE);

    /**
     * One run of rendered output: either ordinary text, or a marker that is encoded as its single
     * special token.
     */
    public record Piece(String text, boolean special) {}

    private Gemma4ToolCalls() {}

    // ---- model input: tool declarations ---------------------------------------------------------

    /**
     * The tool declarations for the system turn — {@code format_function_declaration} for each
     * tool, each wrapped in {@code <|tool>…<tool|>}, with nothing between them.
     *
     * @param toolsJson the tool definitions as the facade builds them: one {@code
     *     {"type":"function","function":{…}}} object per tool, separated by whitespace
     */
    public static List<Piece> renderDeclarations(String toolsJson) {
        Out out = new Out();
        JsonReader reader = new JsonReader(toolsJson);
        reader.skipWhitespace();
        while (!reader.atEnd()) {
            Object tool = reader.value();
            if (tool instanceof Map<?, ?> map) {
                out.special(TOOL_OPEN);
                declaration(out, map);
                out.special(TOOL_CLOSE);
            }
            reader.skipWhitespace();
            while (!reader.atEnd() && reader.peek() == ',') {
                reader.next();
                reader.skipWhitespace();
            }
        }
        return out.pieces();
    }

    /** {@code format_function_declaration}. */
    private static void declaration(Out out, Map<?, ?> tool) {
        Map<?, ?> function = asMap(tool.get("function"));
        if (function == null) {
            function = tool; // a bare function object, without the OpenAI wrapper
        }
        out.text("declaration:" + text(function.get("name")) + "{description:");
        out.quoted(text(function.get("description")));
        Map<?, ?> params = asMap(function.get("parameters"));
        if (params != null && !params.isEmpty()) {
            out.text(",parameters:{");
            Map<?, ?> properties = asMap(params.get("properties"));
            if (properties != null && !properties.isEmpty()) {
                out.text("properties:{");
                parameters(out, properties, false);
                out.text("},");
            }
            List<?> required = asList(params.get("required"));
            if (required != null && !required.isEmpty()) {
                out.text("required:[");
                quotedList(out, required);
                out.text("],");
            }
            if (truthy(params.get("type"))) {
                out.text("type:");
                out.quoted(upper(params.get("type")));
                // The template closes the parameters block only here; a schema without a type
                // is rendered as the template renders it.
                out.text("}");
            }
        }
        Map<?, ?> response = asMap(function.get("response"));
        if (function.containsKey("response")) {
            out.text(",response:{");
            if (response != null && truthy(response.get("description"))) {
                out.text("description:");
                out.quoted(text(response.get("description")));
                out.text(",");
            }
            if (response != null && upper(response.get("type")).equals("OBJECT")) {
                out.text("type:");
                out.quoted(upper(response.get("type")));
                out.text("}");
            }
        }
        out.text("}");
    }

    private static final List<String> STANDARD_KEYS =
            List.of("description", "type", "properties", "required", "nullable");

    /** {@code format_parameters}. */
    private static void parameters(Out out, Map<?, ?> properties, boolean filterKeys) {
        boolean foundFirst = false;
        for (Map.Entry<String, Object> entry : dictsort(properties)) {
            String key = entry.getKey();
            if (filterKeys && STANDARD_KEYS.contains(key)) {
                continue;
            }
            Map<?, ?> value = asMap(entry.getValue());
            if (value == null) {
                value = Map.of();
            }
            if (foundFirst) {
                out.text(",");
            }
            foundFirst = true;
            out.text(key + ":{");
            boolean comma = false;
            if (truthy(value.get("description"))) {
                out.text("description:");
                out.quoted(text(value.get("description")));
                comma = true;
            }
            String type = upper(value.get("type"));
            if (type.equals("STRING")) {
                if (truthy(value.get("enum"))) {
                    comma = comma(out, comma);
                    out.text("enum:");
                    argument(out, value.get("enum"), true);
                }
            } else if (type.equals("ARRAY")) {
                Map<?, ?> items = asMap(value.get("items"));
                if (items != null && !items.isEmpty()) {
                    comma = comma(out, comma);
                    out.text("items:{");
                    boolean itemsFirst = false;
                    for (Map.Entry<String, Object> item : dictsort(items)) {
                        Object itemValue = item.getValue();
                        if (itemValue == JsonReader.NULL) {
                            continue;
                        }
                        if (itemsFirst) {
                            out.text(",");
                        }
                        itemsFirst = true;
                        switch (item.getKey()) {
                            case "properties" -> {
                                out.text("properties:{");
                                Map<?, ?> nested = asMap(itemValue);
                                if (nested != null) {
                                    parameters(out, nested, false);
                                }
                                out.text("}");
                            }
                            case "required" -> {
                                out.text("required:[");
                                List<?> req = asList(itemValue);
                                if (req != null) {
                                    quotedList(out, req);
                                }
                                out.text("]");
                            }
                            case "type" -> {
                                out.text("type:");
                                if (itemValue instanceof String s) {
                                    argument(out, s.toUpperCase(Locale.ROOT), true);
                                } else {
                                    List<Object> upperTypes = new ArrayList<>();
                                    List<?> types = asList(itemValue);
                                    if (types != null) {
                                        for (Object t : types) {
                                            upperTypes.add(upper(t));
                                        }
                                    }
                                    argument(out, upperTypes, true);
                                }
                            }
                            default -> {
                                out.text(item.getKey() + ":");
                                argument(out, itemValue, true);
                            }
                        }
                    }
                    out.text("}");
                }
            }
            if (truthy(value.get("nullable"))) {
                comma = comma(out, comma);
                out.text("nullable:true");
            }
            if (type.equals("OBJECT")) {
                Map<?, ?> nested = asMap(value.get("properties"));
                if (nested != null) {
                    comma = comma(out, comma);
                    out.text("properties:{");
                    parameters(out, nested, false);
                    out.text("}");
                } else {
                    comma = comma(out, comma);
                    out.text("properties:{");
                    parameters(out, value, true);
                    out.text("}");
                }
                List<?> required = asList(value.get("required"));
                if (required != null && !required.isEmpty()) {
                    comma = comma(out, comma);
                    out.text("required:[");
                    quotedList(out, required);
                    out.text("]");
                }
            }
            comma(out, comma);
            out.text("type:");
            out.quoted(type);
            out.text("}");
        }
    }

    /** The template's {@code if add_comma %},{% else %}{% set add_comma = true %}{% endif}. */
    private static boolean comma(Out out, boolean comma) {
        if (comma) {
            out.text(",");
        }
        return true;
    }

    private static void quotedList(Out out, List<?> items) {
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) {
                out.text(",");
            }
            out.quoted(text(items.get(i)));
        }
    }

    // ---- model input: calls and results ---------------------------------------------------------

    /**
     * One call, as the template replays an assistant {@code tool_calls} entry: {@code
     * <|tool_call>call:NAME{key:value,…}<tool_call|>}, keys sorted and bare, values through {@code
     * format_argument}.
     *
     * <p>Arguments that are not a JSON object take the template's pre-serialized branch: an outer
     * {@code {…}} is stripped and the rest written as it stands.
     */
    public static List<Piece> renderCall(ToolCallExtract call) {
        Out out = new Out();
        out.special(CALL_OPEN);
        out.text("call:" + call.name() + "{");
        String raw = call.argumentsJson() == null ? "" : call.argumentsJson().strip();
        Object arguments = null;
        if (!raw.isEmpty()) {
            try {
                arguments = JsonReader.parse(raw);
            } catch (IllegalArgumentException e) {
                arguments = raw; // not JSON: rendered as the pre-serialized string it is
            }
        }
        if (arguments instanceof Map<?, ?> map) {
            boolean first = true;
            for (Map.Entry<String, Object> entry : dictsort(map)) {
                if (!first) {
                    out.text(",");
                }
                first = false;
                out.text(entry.getKey() + ":");
                argument(out, entry.getValue(), false);
            }
        } else if (arguments != null) {
            out.text(
                    raw.startsWith("{") && raw.endsWith("}")
                            ? raw.substring(1, raw.length() - 1)
                            : call.argumentsJson());
        }
        out.text("}");
        out.special(CALL_CLOSE);
        return out.pieces();
    }

    /**
     * One tool result — {@code format_tool_response_block} for a {@code role: tool} message whose
     * content is a string, which is what the facade's opaque result is: {@code
     * <|tool_response>response:NAME{value:<|"|>RESULT<|"|>}<tool_response|>}.
     */
    public static List<Piece> renderResponse(String toolName, String result) {
        Out out = new Out();
        out.special(RESPONSE_OPEN);
        out.text("response:" + toolName + "{value:");
        out.quoted(result);
        out.text("}");
        out.special(RESPONSE_CLOSE);
        return out.pieces();
    }

    /** {@code format_argument}. */
    private static void argument(Out out, Object value, boolean escapeKeys) {
        if (value == null || value == JsonReader.NULL) {
            out.text("null");
        } else if (value instanceof String s) {
            out.quoted(s);
        } else if (value instanceof Boolean b) {
            out.text(b ? "true" : "false");
        } else if (value instanceof Map<?, ?> map) {
            out.text("{");
            boolean first = true;
            for (Map.Entry<String, Object> entry : dictsort(map)) {
                if (!first) {
                    out.text(",");
                }
                first = false;
                if (escapeKeys) {
                    out.quoted(entry.getKey());
                } else {
                    out.text(entry.getKey());
                }
                out.text(":");
                argument(out, entry.getValue(), escapeKeys);
            }
            out.text("}");
        } else if (value instanceof List<?> list) {
            out.text("[");
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) {
                    out.text(",");
                }
                argument(out, list.get(i), escapeKeys);
            }
            out.text("]");
        } else {
            out.text(value.toString()); // a number, as it was written
        }
    }

    // ---- model output → ToolCallExtract ---------------------------------------------------------

    /**
     * Every tool call in a response, in the order the model wrote them.
     *
     * <p>A call counts when its name and its argument dictionary are complete; the closing {@code
     * <tool_call|>} is not required, since generation stops on the token after it. A call whose
     * arguments do not parse is left out rather than reported: a caller would execute it.
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
            int next = responseText.indexOf(CALL_OPEN, bodyStart);
            if (end != -1 && (next == -1 || end < next)) {
                parseOne(responseText.substring(bodyStart, end)).ifPresent(calls::add);
                from = end + CALL_CLOSE.length();
            } else {
                // Unclosed: the call runs to the next one, or to the end of the response.
                int bodyEnd = next == -1 ? responseText.length() : next;
                parseOne(responseText.substring(bodyStart, bodyEnd)).ifPresent(calls::add);
                from = bodyEnd;
            }
        }
        return calls;
    }

    /** The first tool call in a response, or empty. */
    public static Optional<ToolCallExtract> parseFirst(String responseText) {
        List<ToolCallExtract> all = parseAll(responseText);
        return all.isEmpty() ? Optional.empty() : Optional.of(all.get(0));
    }

    /** {@code call:NAME{…}}, without the surrounding markers. */
    private static Optional<ToolCallExtract> parseOne(String body) {
        String s = body.strip();
        if (s.startsWith("call:")) {
            s = s.substring("call:".length());
        }
        int brace = s.indexOf('{');
        if (brace <= 0) {
            return Optional.empty();
        }
        String name = s.substring(0, brace).strip();
        if (name.isEmpty()) {
            return Optional.empty();
        }
        DslReader reader = new DslReader(s, brace);
        try {
            StringBuilder json = new StringBuilder();
            reader.dict(json);
            return Optional.of(new ToolCallExtract(name, json.toString()));
        } catch (IllegalArgumentException e) {
            return Optional.empty();
        }
    }

    /** Reads the dictionary syntax and writes the equivalent JSON. */
    private static final class DslReader {
        private final String s;
        private int i;

        DslReader(String s, int start) {
            this.s = s;
            this.i = start;
        }

        void dict(StringBuilder json) {
            expect('{');
            json.append('{');
            ws();
            if (peek() == '}') {
                i++;
                json.append('}');
                return;
            }
            boolean first = true;
            while (true) {
                ws();
                int keyStart = i;
                while (i < s.length() && s.charAt(i) != ':' && s.charAt(i) != '}') {
                    i++;
                }
                String key = s.substring(keyStart, i).strip();
                if (key.startsWith(QUOTE)
                        && key.endsWith(QUOTE)
                        && key.length() >= 2 * QUOTE.length()) {
                    key = key.substring(QUOTE.length(), key.length() - QUOTE.length());
                }
                expect(':');
                if (key.isEmpty()) {
                    throw new IllegalArgumentException("empty key");
                }
                if (!first) {
                    json.append(',');
                }
                first = false;
                json.append(JsonReader.quote(key)).append(':');
                ws();
                value(json);
                ws();
                char c = peek();
                if (c == ',') {
                    i++;
                } else if (c == '}') {
                    i++;
                    json.append('}');
                    return;
                } else {
                    throw new IllegalArgumentException("unterminated dictionary");
                }
            }
        }

        void array(StringBuilder json) {
            expect('[');
            json.append('[');
            ws();
            if (peek() == ']') {
                i++;
                json.append(']');
                return;
            }
            boolean first = true;
            while (true) {
                ws();
                if (!first) {
                    json.append(',');
                }
                first = false;
                value(json);
                ws();
                char c = peek();
                if (c == ',') {
                    i++;
                } else if (c == ']') {
                    i++;
                    json.append(']');
                    return;
                } else {
                    throw new IllegalArgumentException("unterminated array");
                }
            }
        }

        void value(StringBuilder json) {
            if (s.startsWith(QUOTE, i)) {
                int start = i + QUOTE.length();
                int end = s.indexOf(QUOTE, start);
                if (end == -1) {
                    throw new IllegalArgumentException("unterminated string");
                }
                json.append(JsonReader.quote(s.substring(start, end)));
                i = end + QUOTE.length();
                return;
            }
            char c = peek();
            if (c == '{') {
                dict(json);
            } else if (c == '[') {
                array(json);
            } else if (c == '"') {
                // An ordinary JSON string, which the model occasionally writes instead.
                int start = i;
                i++;
                while (i < s.length() && s.charAt(i) != '"') {
                    if (s.charAt(i) == '\\') {
                        i++;
                    }
                    i++;
                }
                if (i >= s.length()) {
                    throw new IllegalArgumentException("unterminated string");
                }
                i++;
                json.append(s, start, i);
            } else {
                int start = i;
                while (i < s.length() && ",}]".indexOf(s.charAt(i)) == -1) {
                    i++;
                }
                if (i >= s.length()) {
                    throw new IllegalArgumentException("unterminated value");
                }
                String bare = s.substring(start, i).strip();
                if (bare.equals("true") || bare.equals("false") || bare.equals("null")) {
                    json.append(bare);
                } else if (JsonReader.isNumber(bare)) {
                    json.append(bare);
                } else if (bare.isEmpty()) {
                    throw new IllegalArgumentException("missing value");
                } else {
                    json.append(JsonReader.quote(bare)); // an unquoted string
                }
            }
        }

        void ws() {
            while (i < s.length() && Character.isWhitespace(s.charAt(i))) {
                i++;
            }
        }

        char peek() {
            if (i >= s.length()) {
                throw new IllegalArgumentException("unexpected end");
            }
            return s.charAt(i);
        }

        void expect(char c) {
            if (peek() != c) {
                throw new IllegalArgumentException("expected '" + c + "' at " + i);
            }
            i++;
        }
    }

    // ---- helpers --------------------------------------------------------------------------------

    /** Jinja's {@code dictsort}: by key, case-insensitively, ties in insertion order. */
    @SuppressWarnings("unchecked")
    private static List<Map.Entry<String, Object>> dictsort(Map<?, ?> map) {
        List<Map.Entry<String, Object>> entries =
                new ArrayList<>(((Map<String, Object>) map).entrySet());
        entries.sort(Comparator.comparing(e -> e.getKey().toLowerCase(Locale.ROOT)));
        return entries;
    }

    private static Map<?, ?> asMap(Object value) {
        return value instanceof Map<?, ?> map ? map : null;
    }

    private static List<?> asList(Object value) {
        return value instanceof List<?> list ? list : null;
    }

    /** Jinja truthiness, for the values a schema holds. */
    private static boolean truthy(Object value) {
        if (value == null || value == JsonReader.NULL) {
            return false;
        }
        if (value instanceof String s) {
            return !s.isEmpty();
        }
        if (value instanceof Boolean b) {
            return b;
        }
        if (value instanceof Map<?, ?> m) {
            return !m.isEmpty();
        }
        if (value instanceof List<?> l) {
            return !l.isEmpty();
        }
        if (value instanceof JsonReader.Number n) {
            return Double.parseDouble(n.text()) != 0;
        }
        return true;
    }

    /** A value as the template interpolates it; an absent one is the empty string. */
    private static String text(Object value) {
        if (value == null || value == JsonReader.NULL) {
            return "";
        }
        return value.toString();
    }

    private static String upper(Object value) {
        return text(value).toUpperCase(Locale.ROOT);
    }

    /** Collects pieces, merging adjacent text. */
    private static final class Out {
        private final List<Piece> pieces = new ArrayList<>();
        private final StringBuilder text = new StringBuilder();

        void text(String s) {
            text.append(s);
        }

        void special(String marker) {
            flush();
            pieces.add(new Piece(marker, true));
        }

        void quoted(String s) {
            special(QUOTE);
            text(s);
            special(QUOTE);
        }

        List<Piece> pieces() {
            flush();
            return List.copyOf(pieces);
        }

        private void flush() {
            if (!text.isEmpty()) {
                pieces.add(new Piece(text.toString(), false));
                text.setLength(0);
            }
        }
    }

    /** The pieces as one string, markers written out — what the template renders. */
    public static String asText(List<Piece> pieces) {
        StringBuilder sb = new StringBuilder();
        for (Piece piece : pieces) {
            sb.append(piece.text());
        }
        return sb.toString();
    }

    /**
     * A minimal JSON reader, into {@link LinkedHashMap} / {@link List} / {@link String} / {@link
     * Boolean} / {@link Number} / {@link #NULL}. Numbers keep the text they were written with, so
     * {@code 4} is rendered as {@code 4} and not as {@code 4.0}. The project carries no JSON
     * dependency, and the server's reader parses numbers to doubles.
     */
    static final class JsonReader {

        /** JSON {@code null}, distinct from an absent key. */
        static final Object NULL =
                new Object() {
                    @Override
                    public String toString() {
                        return "null";
                    }
                };

        /** A JSON number, as written. */
        record Number(String text) {
            @Override
            public String toString() {
                return text;
            }
        }

        private final String s;
        private int i;

        JsonReader(String s) {
            this.s = s == null ? "" : s;
        }

        static Object parse(String s) {
            JsonReader reader = new JsonReader(s);
            reader.skipWhitespace();
            Object value = reader.value();
            reader.skipWhitespace();
            if (!reader.atEnd()) {
                throw new IllegalArgumentException("trailing characters at " + reader.i);
            }
            return value;
        }

        boolean atEnd() {
            return i >= s.length();
        }

        char peek() {
            if (atEnd()) {
                throw new IllegalArgumentException("unexpected end of JSON");
            }
            return s.charAt(i);
        }

        char next() {
            char c = peek();
            i++;
            return c;
        }

        void skipWhitespace() {
            while (i < s.length() && Character.isWhitespace(s.charAt(i))) {
                i++;
            }
        }

        Object value() {
            skipWhitespace();
            char c = peek();
            switch (c) {
                case '{':
                    return object();
                case '[':
                    return array();
                case '"':
                    return string();
                case 't':
                    literal("true");
                    return Boolean.TRUE;
                case 'f':
                    literal("false");
                    return Boolean.FALSE;
                case 'n':
                    literal("null");
                    return NULL;
                default:
                    return number();
            }
        }

        private Map<String, Object> object() {
            next();
            Map<String, Object> map = new LinkedHashMap<>();
            skipWhitespace();
            if (peek() == '}') {
                next();
                return map;
            }
            while (true) {
                skipWhitespace();
                String key = string();
                skipWhitespace();
                if (next() != ':') {
                    throw new IllegalArgumentException("expected ':' at " + (i - 1));
                }
                map.put(key, value());
                skipWhitespace();
                char c = next();
                if (c == '}') {
                    return map;
                }
                if (c != ',') {
                    throw new IllegalArgumentException("expected ',' or '}' at " + (i - 1));
                }
            }
        }

        private List<Object> array() {
            next();
            List<Object> list = new ArrayList<>();
            skipWhitespace();
            if (peek() == ']') {
                next();
                return list;
            }
            while (true) {
                list.add(value());
                skipWhitespace();
                char c = next();
                if (c == ']') {
                    return list;
                }
                if (c != ',') {
                    throw new IllegalArgumentException("expected ',' or ']' at " + (i - 1));
                }
            }
        }

        private String string() {
            if (next() != '"') {
                throw new IllegalArgumentException("expected a string at " + (i - 1));
            }
            StringBuilder sb = new StringBuilder();
            while (true) {
                char c = next();
                if (c == '"') {
                    return sb.toString();
                }
                if (c != '\\') {
                    sb.append(c);
                    continue;
                }
                char e = next();
                switch (e) {
                    case 'n' -> sb.append('\n');
                    case 'r' -> sb.append('\r');
                    case 't' -> sb.append('\t');
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'u' -> {
                        if (i + 4 > s.length()) {
                            throw new IllegalArgumentException("truncated \\u escape");
                        }
                        sb.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                        i += 4;
                    }
                    default -> sb.append(e);
                }
            }
        }

        private void literal(String word) {
            if (!s.startsWith(word, i)) {
                throw new IllegalArgumentException("unexpected token at " + i);
            }
            i += word.length();
        }

        private Number number() {
            int start = i;
            while (i < s.length() && "+-0123456789.eE".indexOf(s.charAt(i)) >= 0) {
                i++;
            }
            String text = s.substring(start, i);
            if (!isNumber(text)) {
                throw new IllegalArgumentException("not a JSON value at " + start);
            }
            return new Number(text);
        }

        static boolean isNumber(String v) {
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

        static String quote(String text) {
            return Qwen35ToolCalls.quote(text);
        }
    }
}
