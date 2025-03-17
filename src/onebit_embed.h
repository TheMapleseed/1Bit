#ifndef ONEBIT_EMBED_H
#define ONEBIT_EMBED_H

#include "onebit_memory.h"
#include "onebit_quant.h"

// Embedding configuration
typedef struct {
    int vocab_size;
    int hidden_size;
    int max_position;
    float dropout_prob;
} EmbeddingConfig;

// Embedding layer
typedef struct {
    QuantizedTensor word_embeddings;
    QuantizedTensor position_embeddings;
    float* embedding_output;
    OneBitMemoryPool* pool;
    EmbeddingConfig config;
} EmbeddingLayer;

// Function declarations
int init_embedding_layer(EmbeddingLayer* embed, const EmbeddingConfig* config,
                        OneBitMemoryPool* pool);
void forward_embedding(EmbeddingLayer* embed, const int* input_ids,
                      float* output, int seq_length);
void cleanup_embedding_layer(EmbeddingLayer* embed);

// Helper functions
void apply_word_embeddings(const QuantizedTensor* embeddings,
                          const int* input_ids, float* output,
                          int seq_length, int hidden_size);
void apply_position_embeddings(const QuantizedTensor* embeddings,
                             float* output, int seq_length,
                             int hidden_size);
void apply_embedding_dropout(float* embeddings, int size,
                           float dropout_prob, unsigned int* seed);

#endif 