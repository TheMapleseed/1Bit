#ifndef ONEBIT_ATTENTION_H
#define ONEBIT_ATTENTION_H

#include "onebit_memory.h"
#include "onebit_quant.h"

// Attention configuration
typedef struct {
    int hidden_size;
    int num_heads;
    int head_dim;
    float attention_dropout;
    int max_seq_length;
} AttentionConfig;

// Attention layer
typedef struct {
    QuantizedTensor query_weight;
    QuantizedTensor key_weight;
    QuantizedTensor value_weight;
    QuantizedTensor output_weight;
    float* query_bias;
    float* key_bias;
    float* value_bias;
    float* output_bias;
    float* attention_scores;
    float* attention_probs;
    float* attention_output;
    OneBitMemoryPool* pool;
    AttentionConfig config;
} AttentionLayer;

// Function declarations
int init_attention_layer(AttentionLayer* layer, const AttentionConfig* config,
                        OneBitMemoryPool* pool);
void forward_attention(AttentionLayer* layer, const float* input,
                      float* output, int seq_length);
void cleanup_attention_layer(AttentionLayer* layer);

// Helper functions
void compute_qkv(AttentionLayer* layer, const float* input,
                 float* query, float* key, float* value, int seq_length);
void compute_attention_scores(AttentionLayer* layer, const float* query,
                            const float* key, int seq_length);
void apply_attention_mask(AttentionLayer* layer, const int* attention_mask,
                         int seq_length);
void compute_attention_output(AttentionLayer* layer, const float* value,
                            float* output, int seq_length);

#endif 