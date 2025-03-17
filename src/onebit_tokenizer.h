#ifndef ONEBIT_TOKENIZER_H
#define ONEBIT_TOKENIZER_H

#include <stdint.h>
#include <stdbool.h>

// Tokenizer types
typedef enum {
    TOKENIZER_BPE,
    TOKENIZER_WORDPIECE,
    TOKENIZER_UNIGRAM,
    TOKENIZER_SENTENCEPIECE
} TokenizerType;

// Token
typedef struct {
    uint32_t id;
    char* text;
    float score;
    bool is_special;
} Token;

// Tokenizer context
typedef struct {
    TokenizerType type;
    char* vocab_file;
    Token* vocab;
    size_t vocab_size;
    bool add_special_tokens;
    void* tokenizer_state;
} TokenizerContext;

// Function declarations
int tokenizer_init(TokenizerContext* ctx, const char* vocab_file, TokenizerType type);
void tokenizer_cleanup(TokenizerContext* ctx);

// Core tokenization
int tokenizer_encode(TokenizerContext* ctx, const char* text, 
                    uint32_t* token_ids, size_t* num_tokens);
int tokenizer_decode(TokenizerContext* ctx, const uint32_t* token_ids,
                    size_t num_tokens, char* text, size_t* text_len);

// Batch processing
int tokenizer_encode_batch(TokenizerContext* ctx, const char** texts,
                          size_t num_texts, uint32_t** token_ids,
                          size_t** lengths);

// Vocabulary operations
int tokenizer_add_tokens(TokenizerContext* ctx, const char** tokens,
                        size_t num_tokens);
int tokenizer_save(TokenizerContext* ctx, const char* path);
int tokenizer_load(TokenizerContext* ctx, const char* path);

#endif 