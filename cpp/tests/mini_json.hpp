/// \file mini_json.hpp
/// \brief Tiny self-contained JSON reader for the flat golden.json schema.
///
/// Supports objects, arrays, strings (basic escapes), numbers, booleans
/// and null — everything the golden file needs, nothing more.  Test-tree
/// only; the etrade library itself does no file I/O.  Numbers are read in
/// the classic "C" locale whatever the global locale is, nesting is capped
/// at kMaxDepth, and a `\u` escape must carry four hex digits (decoded to
/// UTF-8; surrogate pairs are not combined).
#pragma once

#include <cctype>
#include <fstream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mini_json {

/// Deepest array/object nesting accepted by the parser.
inline constexpr int kMaxDepth = 64;

/// A parsed JSON value (tagged union over the vector/obj members).
struct Value {
    enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };
    Type type = Type::kNull;
    bool boolean = false;
    double number = 0.0;
    std::string str;
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> obj;

    /// Object member lookup; throws when the key is missing.
    const Value& at(const std::string& key) const {
        for (const auto& kv : obj) {
            if (kv.first == key) return kv.second;
        }
        throw std::runtime_error("mini_json: missing key '" + key + "'");
    }

    /// True when the object holds `key`.
    bool has(const std::string& key) const {
        for (const auto& kv : obj) {
            if (kv.first == key) return true;
        }
        return false;
    }
};

namespace detail {

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}

    Value parse() {
        Value v = parse_value();
        skip_ws();
        if (pos_ != text_.size()) fail("trailing content");
        return v;
    }

private:
    const std::string& text_;
    std::size_t pos_ = 0;
    int depth_ = 0;

    /// RAII nesting counter for arrays and objects.
    class DepthGuard {
    public:
        explicit DepthGuard(Parser& parser) : parser_(parser) {
            if (++parser_.depth_ > kMaxDepth) {
                parser_.fail("nesting deeper than " + std::to_string(kMaxDepth));
            }
        }
        ~DepthGuard() { --parser_.depth_; }
        DepthGuard(const DepthGuard&) = delete;
        DepthGuard& operator=(const DepthGuard&) = delete;

    private:
        Parser& parser_;
    };

    [[noreturn]] void fail(const std::string& why) const {
        throw std::runtime_error("mini_json: " + why + " at offset " +
                                 std::to_string(pos_));
    }

    static int hex_digit(char h) {
        if (h >= '0' && h <= '9') return h - '0';
        if (h >= 'a' && h <= 'f') return h - 'a' + 10;
        if (h >= 'A' && h <= 'F') return h - 'A' + 10;
        return -1;
    }

    static void append_utf8(std::string& out, unsigned cp) {
        if (cp < 0x80U) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800U) {
            out += static_cast<char>(0xC0U | (cp >> 6U));
            out += static_cast<char>(0x80U | (cp & 0x3FU));
        } else {
            out += static_cast<char>(0xE0U | (cp >> 12U));
            out += static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU));
            out += static_cast<char>(0x80U | (cp & 0x3FU));
        }
    }

    void skip_ws() {
        while (pos_ < text_.size() &&
               (text_[pos_] == ' ' || text_[pos_] == '\t' ||
                text_[pos_] == '\n' || text_[pos_] == '\r')) {
            ++pos_;
        }
    }

    char peek() {
        if (pos_ >= text_.size()) fail("unexpected end of input");
        return text_[pos_];
    }

    void expect(char c) {
        if (peek() != c) fail(std::string("expected '") + c + "'");
        ++pos_;
    }

    bool consume_literal(const char* lit) {
        const std::size_t n = std::string(lit).size();
        if (text_.compare(pos_, n, lit) == 0) {
            pos_ += n;
            return true;
        }
        return false;
    }

    Value parse_value() {
        skip_ws();
        const char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') {
            Value v;
            v.type = Value::Type::kString;
            v.str = parse_string();
            return v;
        }
        if (consume_literal("true")) {
            Value v;
            v.type = Value::Type::kBool;
            v.boolean = true;
            return v;
        }
        if (consume_literal("false")) {
            Value v;
            v.type = Value::Type::kBool;
            v.boolean = false;
            return v;
        }
        if (consume_literal("null")) return Value{};
        return parse_number();
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (pos_ >= text_.size()) fail("unterminated string");
            const char c = text_[pos_++];
            if (c == '"') break;
            if (c == '\\') {
                if (pos_ >= text_.size()) fail("bad escape");
                const char e = text_[pos_++];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        if (pos_ + 4 > text_.size()) fail("truncated \\u escape");
                        unsigned cp = 0;
                        for (int i = 0; i < 4; ++i) {
                            const int d = hex_digit(text_[pos_]);
                            if (d < 0) fail("malformed \\u escape (expected 4 hex digits)");
                            cp = cp * 16U + static_cast<unsigned>(d);
                            ++pos_;
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default: fail("unknown escape");
                }
            } else {
                out += c;
            }
        }
        return out;
    }

    Value parse_number() {
        const std::size_t start = pos_;
        while (pos_ < text_.size() &&
               (std::isdigit(static_cast<unsigned char>(text_[pos_])) ||
                text_[pos_] == '-' || text_[pos_] == '+' || text_[pos_] == '.' ||
                text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
        }
        if (pos_ == start) fail("invalid value");
        const std::string tok = text_.substr(start, pos_ - start);
        // Locale-independent: a global locale with ',' as the decimal mark
        // must not change how the golden file reads.
        std::istringstream in(tok);
        in.imbue(std::locale::classic());
        double v = 0.0;
        in >> v;
        if (in.fail() || in.peek() != std::char_traits<char>::eof()) {
            fail("invalid number '" + tok + "'");
        }
        Value out;
        out.type = Value::Type::kNumber;
        out.number = v;
        return out;
    }

    Value parse_array() {
        const DepthGuard guard(*this);
        expect('[');
        Value v;
        v.type = Value::Type::kArray;
        skip_ws();
        if (peek() == ']') {
            ++pos_;
            return v;
        }
        while (true) {
            v.arr.push_back(parse_value());
            skip_ws();
            const char c = peek();
            ++pos_;
            if (c == ']') break;
            if (c != ',') fail("expected ',' or ']'");
        }
        return v;
    }

    Value parse_object() {
        const DepthGuard guard(*this);
        expect('{');
        Value v;
        v.type = Value::Type::kObject;
        skip_ws();
        if (peek() == '}') {
            ++pos_;
            return v;
        }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            v.obj.emplace_back(std::move(key), parse_value());
            skip_ws();
            const char c = peek();
            ++pos_;
            if (c == '}') break;
            if (c != ',') fail("expected ',' or '}'");
        }
        return v;
    }
};

}  // namespace detail

/// Parse a JSON document from a string.
inline Value parse(const std::string& text) { return detail::Parser(text).parse(); }

/// Parse a JSON document from a file; throws std::runtime_error when the
/// file cannot be read.
inline Value parse_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("mini_json: cannot open " + path);
    std::stringstream buf;
    buf << in.rdbuf();
    return parse(buf.str());
}

}  // namespace mini_json
