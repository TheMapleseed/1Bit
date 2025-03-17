#include "onebit/onebit_tokenizer.h"
#include "onebit/onebit_error.h"
#include "onebit/onebit_hash.h"
#include "onebit/onebit_string.h"
#include <utf8proc.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Internal token entry structure
typedef struct TokenEntry {
    char* text;
    TokenType type;
    uint32_t id;
    struct TokenEntry* next;
} TokenEntry;

// Internal tokenizer context structure
struct TokenizerContext {
    TokenEntry** vocab;         // Hash table for vocabulary
    size_t vocab_size;         // Current vocabulary size
    size_t max_vocab_size;     // Maximum vocabulary size
    size_t hash_size;          // Size of hash table
    bool case_sensitive;       // Case sensitivity flag
    bool normalize_unicode;    // Unicode normalization flag
    char* special_tokens;      // Special tokens string
};

// Hash function for vocabulary lookup
static size_t hash_token(const char* text, size_t hash_size) {
    return hash_string(text) % hash_size;
}

// Create new token entry
static TokenEntry* create_token_entry(const char* text, TokenType type, uint32_t id) {
    TokenEntry* entry = malloc(sizeof(TokenEntry));
    if (!entry) return NULL;
    
    entry->text = strdup(text);
    if (!entry->text) {
        free(entry);
        return NULL;
    }
    
    entry->type = type;
    entry->id = id;
    entry->next = NULL;
    return entry;
}

int tokenizer_init(TokenizerContext** ctx, const TokenizerConfig* config) {
    if (!ctx || !config) return ONEBIT_ERROR_INVALID_PARAM;
    
    *ctx = malloc(sizeof(TokenizerContext));
    if (!*ctx) return ONEBIT_ERROR_MEMORY;
    
    TokenizerContext* context = *ctx;
    context->max_vocab_size = config->max_vocab_size;
    context->hash_size = context->max_vocab_size * 2;  // Load factor 0.5
    context->vocab_size = 0;
    context->case_sensitive = config->case_sensitive;
    context->normalize_unicode = config->normalize_unicode;
    
    // Allocate vocabulary hash table
    context->vocab = calloc(context->hash_size, sizeof(TokenEntry*));
    if (!context->vocab) {
        free(context);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Copy special tokens if provided
    if (config->special_tokens) {
        context->special_tokens = strdup(config->special_tokens);
        if (!context->special_tokens) {
            free(context->vocab);
            free(context);
            return ONEBIT_ERROR_MEMORY;
        }
    } else {
        context->special_tokens = NULL;
    }
    
    // Load vocabulary if provided
    if (config->vocab_file) {
        int result = tokenizer_load_vocab(context, config->vocab_file);
        if (result != ONEBIT_SUCCESS) {
            tokenizer_cleanup(context);
            return result;
        }
    }
    
    return ONEBIT_SUCCESS;
}

void tokenizer_cleanup(TokenizerContext* ctx) {
    if (!ctx) return;
    
    // Free vocabulary entries
    for (size_t i = 0; i < ctx->hash_size; i++) {
        TokenEntry* entry = ctx->vocab[i];
        while (entry) {
            TokenEntry* next = entry->next;
            free(entry->text);
            free(entry);
            entry = next;
        }
    }
    
    free(ctx->vocab);
    free(ctx->special_tokens);
    free(ctx);
}

static TokenType detect_token_type(const char* text) {
    if (!text || !*text) return TOKEN_UNKNOWN;
    
    // Check first character
    if (isspace(*text)) return TOKEN_WHITESPACE;
    if (isdigit(*text)) return TOKEN_NUMBER;
    if (ispunct(*text)) return TOKEN_PUNCTUATION;
    if (isalpha(*text)) return TOKEN_WORD;
    
    return TOKEN_UNKNOWN;
}

static int normalize_text(const char* input, char* output, size_t output_size,
                        bool case_sensitive, bool normalize_unicode) {
    if (!input || !output || output_size == 0) return ONEBIT_ERROR_INVALID_PARAM;
    
    utf8proc_uint8_t* normalized = NULL;
    
    if (normalize_unicode) {
        normalized = utf8proc_NFKC((const utf8proc_uint8_t*)input);
        if (!normalized) return ONEBIT_ERROR_UNICODE;
        input = (const char*)normalized;
    }
    
    size_t len = strlen(input);
    if (len >= output_size) {
        free(normalized);
        return ONEBIT_ERROR_BUFFER_SIZE;
    }
    
    if (case_sensitive) {
        strcpy(output, input);
    } else {
        for (size_t i = 0; i <= len; i++) {
            output[i] = tolower(input[i]);
        }
    }
    
    free(normalized);
    return ONEBIT_SUCCESS;
}

int tokenizer_encode(TokenizerContext* ctx, const char* text,
                    Token* tokens, size_t* num_tokens) {
    if (!ctx || !text || !tokens || !num_tokens) return ONEBIT_ERROR_INVALID_PARAM;
    
    // Normalize input text
    char* normalized = normalize_text(text);
    if (!normalized) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    size_t tokens_added = 0;
    char* token_start = normalized;
    
    while (*token_start && tokens_added < ctx->max_vocab_size) {
        // Find token in vocabulary
        bool found = false;
        size_t longest_match = 0;
        uint32_t longest_match_id = 0;
        
        for (size_t i = 0; i < ctx->vocab_size; i++) {
            size_t match_len = strncmp(token_start, ctx->vocab[i].token, strlen(ctx->vocab[i].token));
            if (match_len == 0 && strlen(ctx->vocab[i].token) > longest_match) {
                found = true;
                longest_match = strlen(ctx->vocab[i].token);
                longest_match_id = ctx->vocab[i].id;
            }
        }
        
        if (found) {
            tokens[tokens_added++] = longest_match_id;
            token_start += longest_match;
        } else {
            token_start++;
        }
    }
    
    *num_tokens = tokens_added;
    return ONEBIT_SUCCESS;
} 