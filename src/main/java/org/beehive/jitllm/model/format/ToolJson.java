package org.beehive.jitllm.model.format;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A minimal JSON reader, into {@link LinkedHashMap} / {@link List} / {@link String} / {@link
 * Boolean} / {@link Number} / {@link #NULL}. Numbers keep the text they were written with, so
 * {@code 4} is rendered as {@code 4} and not as {@code 4.0}. The project carries no JSON
 * dependency, and the server's reader parses numbers to doubles.
 */
final class ToolJson {

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

    ToolJson(String s) {
        this.s = s == null ? "" : s;
    }

    static Object parse(String s) {
        ToolJson reader = new ToolJson(s);
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

    /**
     * Every JSON value in a text that holds several, separated by whitespace or commas — the shape
     * the facade gives the tool definitions in.
     */
    static List<Object> parseSequence(String text) {
        List<Object> values = new ArrayList<>();
        ToolJson reader = new ToolJson(text);
        reader.skipWhitespace();
        while (!reader.atEnd()) {
            values.add(reader.value());
            reader.skipWhitespace();
            while (!reader.atEnd() && reader.peek() == ',') {
                reader.next();
                reader.skipWhitespace();
            }
        }
        return values;
    }

    /**
     * Python's {@code json.dumps(value)} with its default separators ({@code ", "} and {@code ":
     * "}) and {@code ensure_ascii=False} — what a chat template's {@code tojson} writes. Key order
     * is the order the value was read in.
     */
    static String dumps(Object value) {
        StringBuilder out = new StringBuilder();
        write(out, value, -1, 0);
        return out.toString();
    }

    /** Python's {@code json.dumps(value, indent=indent)}: {@code tojson(indent=…)}. */
    static String dumps(Object value, int indent) {
        StringBuilder out = new StringBuilder();
        write(out, value, indent, 0);
        return out.toString();
    }

    private static void write(StringBuilder out, Object value, int indent, int depth) {
        if (value == null || value == NULL) {
            out.append("null");
        } else if (value instanceof String s) {
            out.append(quote(s));
        } else if (value instanceof Boolean b) {
            out.append(b ? "true" : "false");
        } else if (value instanceof Map<?, ?> map) {
            if (map.isEmpty()) {
                out.append("{}");
                return;
            }
            out.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                separator(out, first, indent, depth + 1);
                first = false;
                out.append(quote(entry.getKey().toString())).append(": ");
                write(out, entry.getValue(), indent, depth + 1);
            }
            newline(out, indent, depth);
            out.append('}');
        } else if (value instanceof List<?> list) {
            if (list.isEmpty()) {
                out.append("[]");
                return;
            }
            out.append('[');
            boolean first = true;
            for (Object item : list) {
                separator(out, first, indent, depth + 1);
                first = false;
                write(out, item, indent, depth + 1);
            }
            newline(out, indent, depth);
            out.append(']');
        } else {
            out.append(value); // a number, as written
        }
    }

    private static void separator(StringBuilder out, boolean first, int indent, int depth) {
        if (indent < 0) {
            if (!first) {
                out.append(", ");
            }
            return;
        }
        if (!first) {
            out.append(',');
        }
        newline(out, indent, depth);
    }

    private static void newline(StringBuilder out, int indent, int depth) {
        if (indent >= 0) {
            out.append('\n').append(" ".repeat(indent * depth));
        }
    }
}
