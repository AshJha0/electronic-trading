package com.quant.etrade;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal recursive-descent JSON parser for the flat golden.json schema
 * (objects, arrays, strings, numbers, booleans, null). Test-tree only.
 */
final class Json {

    private final String src;
    private int pos;

    private Json(String src) {
        this.src = src;
        this.pos = 0;
    }

    /**
     * Parses a JSON document.
     *
     * @param text JSON text
     * @return Map&lt;String,Object&gt;, List&lt;Object&gt;, String, Double, Boolean or null
     */
    static Object parse(String text) {
        Json p = new Json(text);
        p.skipWs();
        Object v = p.value();
        p.skipWs();
        if (p.pos != text.length()) {
            throw new IllegalArgumentException("trailing content at offset " + p.pos);
        }
        return v;
    }

    private Object value() {
        char c = peek();
        switch (c) {
            case '{':
                return object();
            case '[':
                return array();
            case '"':
                return string();
            case 't':
                expect("true");
                return Boolean.TRUE;
            case 'f':
                expect("false");
                return Boolean.FALSE;
            case 'n':
                expect("null");
                return null;
            default:
                return number();
        }
    }

    private Map<String, Object> object() {
        Map<String, Object> out = new LinkedHashMap<>();
        consume('{');
        skipWs();
        if (peek() == '}') {
            pos++;
            return out;
        }
        while (true) {
            skipWs();
            String key = string();
            skipWs();
            consume(':');
            skipWs();
            out.put(key, value());
            skipWs();
            char c = next();
            if (c == '}') {
                return out;
            }
            if (c != ',') {
                throw new IllegalArgumentException("expected ',' or '}' at offset " + (pos - 1));
            }
        }
    }

    private List<Object> array() {
        List<Object> out = new ArrayList<>();
        consume('[');
        skipWs();
        if (peek() == ']') {
            pos++;
            return out;
        }
        while (true) {
            skipWs();
            out.add(value());
            skipWs();
            char c = next();
            if (c == ']') {
                return out;
            }
            if (c != ',') {
                throw new IllegalArgumentException("expected ',' or ']' at offset " + (pos - 1));
            }
        }
    }

    private String string() {
        consume('"');
        StringBuilder sb = new StringBuilder();
        while (true) {
            char c = next();
            if (c == '"') {
                return sb.toString();
            }
            if (c == '\\') {
                char e = next();
                switch (e) {
                    case '"' -> sb.append('"');
                    case '\\' -> sb.append('\\');
                    case '/' -> sb.append('/');
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'n' -> sb.append('\n');
                    case 'r' -> sb.append('\r');
                    case 't' -> sb.append('\t');
                    case 'u' -> {
                        sb.append((char) Integer.parseInt(src.substring(pos, pos + 4), 16));
                        pos += 4;
                    }
                    default -> throw new IllegalArgumentException("bad escape \\" + e);
                }
            } else {
                sb.append(c);
            }
        }
    }

    private Double number() {
        int start = pos;
        while (pos < src.length() && "+-0123456789.eE".indexOf(src.charAt(pos)) >= 0) {
            pos++;
        }
        if (pos == start) {
            throw new IllegalArgumentException("unexpected character at offset " + pos);
        }
        return Double.parseDouble(src.substring(start, pos));
    }

    private char peek() {
        if (pos >= src.length()) {
            throw new IllegalArgumentException("unexpected end of input");
        }
        return src.charAt(pos);
    }

    private char next() {
        char c = peek();
        pos++;
        return c;
    }

    private void consume(char c) {
        if (next() != c) {
            throw new IllegalArgumentException("expected '" + c + "' at offset " + (pos - 1));
        }
    }

    private void expect(String lit) {
        if (!src.startsWith(lit, pos)) {
            throw new IllegalArgumentException("expected '" + lit + "' at offset " + pos);
        }
        pos += lit.length();
    }

    private void skipWs() {
        while (pos < src.length() && Character.isWhitespace(src.charAt(pos))) {
            pos++;
        }
    }
}
