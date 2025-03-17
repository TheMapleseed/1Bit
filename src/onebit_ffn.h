#ifndef ONEBIT_FFN_H
#define ONEBIT_FFN_H

#include "onebit_memory.h"
#include "onebit_quant.h"

// FFN configuration
typedef struct {
    int hidden_size;
    int intermediate_size;
    float dropout_prob;
    char* activation_fn;  // "gelu" or "relu"
} FFNConfig;

// FFN layer
typedef struct {
    QuantizedTensor fc1_weight;
    QuantizedTensor fc2_weight;
    float* fc1_bias;
    float* fc2_bias;
    float* intermediate;
    OneBitMemoryPool* pool;
    FFNConfig config;
} FFNLayer;

// Function declarations
int init_ffn_layer(FFNLayer* layer, const FFNConfig* config,
                  OneBitMemoryPool* pool);
void forward_ffn(FFNLayer* layer, const float* input,
                float* output, int seq_length);
void cleanup_ffn_layer(FFNLayer* layer);

// Helper functions
void linear_forward(const QuantizedTensor* weight, const float* bias,
                   const float* input, float* output,
                   int batch_size, int in_features, int out_features);
void apply_dropout(float* data, int size, float prob,
                  unsigned int* seed);

#endif 