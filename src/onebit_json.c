#include "onebit/onebit_json.h"
#include "onebit/onebit_error.h"
#include <string.h>
#include <ctype.h>

// JSON value types
typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} JsonType;

// JSON value
typedef struct JsonValue {
    JsonType type;
    union {
        bool bool_value;
        double number_value;
        char* string_value;
        struct {
            struct JsonValue** items;
            size_t size;
            size_t capacity;
        } array;
        struct {
            char** keys;
            struct JsonValue** values;
            size_t size;
            size_t capacity;
        } object;
    };
} JsonValue;

// Parser state
typedef struct {
    const char* input;
    size_t pos;
    size_t len;
} JsonParser;

static void skip_whitespace(JsonParser* parser) {
    while (parser->pos < parser->len &&
           isspace(parser->input[parser->pos])) {
        parser->pos++;
    }
}

static bool parse_literal(JsonParser* parser, const char* literal) {
    size_t len = strlen(literal);
    if (parser->pos + len > parser->len) {
        return false;
    }
    
    if (strncmp(parser->input + parser->pos, literal, len) == 0) {
        parser->pos += len;
        return true;
    }
    
    return false;
}

static JsonValue* parse_value(JsonParser* parser);

static JsonValue* parse_string(JsonParser* parser) {
    if (parser->pos >= parser->len ||
        parser->input[parser->pos] != '"') {
        return NULL;
    }
    
    parser->pos++; // Skip opening quote
    
    // Find closing quote
    size_t start = parser->pos;
    while (parser->pos < parser->len) {
        if (parser->input[parser->pos] == '"' &&
            parser->input[parser->pos - 1] != '\\') {
            break;
        }
        parser->pos++;
    }
    
    if (parser->pos >= parser->len) {
        return NULL;
    }
    
    size_t len = parser->pos - start;
    char* str = malloc(len + 1);
    if (!str) {
        return NULL;
    }
    
    memcpy(str, parser->input + start, len);
    str[len] = '\0';
    
    JsonValue* value = malloc(sizeof(JsonValue));
    if (!value) {
        free(str);
        return NULL;
    }
    
    value->type = JSON_STRING;
    value->string_value = str;
    
    return value;
} 