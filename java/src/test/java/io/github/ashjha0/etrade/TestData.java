package io.github.ashjha0.etrade;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Loads the shared golden/reference JSON files (Jackson). The data root is the
 * {@code etrade.data} system property (surefire sets it), defaulting to {@code ../data}.
 */
final class TestData {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private TestData() {
    }

    /** @return the data root directory */
    static Path root() {
        return Path.of(System.getProperty("etrade.data", "../data"));
    }

    /**
     * Parses a JSON file under the data root into plain Java values: objects are
     * {@code Map<String,Object>} (insertion ordered), arrays {@code List<Object>},
     * every number a {@code Double}, plus {@code String}, {@code Boolean} and null.
     *
     * @param rel path relative to the data root, e.g. {@code "golden/golden.json"}
     * @return the parsed document
     * @throws IOException when the file cannot be read or parsed
     */
    static Object load(String rel) throws IOException {
        return convert(MAPPER.readTree(root().resolve(rel).toFile()));
    }

    private static Object convert(JsonNode n) {
        if (n.isObject()) {
            Map<String, Object> m = new LinkedHashMap<>();
            n.properties().forEach(e -> m.put(e.getKey(), convert(e.getValue())));
            return m;
        }
        if (n.isArray()) {
            List<Object> l = new ArrayList<>(n.size());
            n.forEach(c -> l.add(convert(c)));
            return l;
        }
        if (n.isNumber()) {
            return n.doubleValue();
        }
        if (n.isTextual()) {
            return n.textValue();
        }
        if (n.isBoolean()) {
            return n.booleanValue();
        }
        return null;
    }
}
