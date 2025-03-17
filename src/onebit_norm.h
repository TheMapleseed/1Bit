#ifndef ONEBIT_NORM_H
#define ONEBIT_NORM_H

#include "onebit_memory.h"

// Layer normalization configuration
typedef struct {
    int hidden_size;
    float eps;
} LayerNormConfig;

// Layer normalization layer
typedef struct {
    float* weight;
    float* bias;
    float* mean;
    float* variance;
    OneBitMemoryPool* pool;
    LayerNormConfig config;
} LayerNorm;

// Function declarations
int init_layer_norm(LayerNorm* norm, const LayerNormConfig* config,
                   OneBitMemoryPool* pool);
void forward_layer_norm(LayerNorm* norm, const float* input,
                       float* output, int seq_length);
void cleanup_layer_norm(LayerNorm* norm);

// Helper functions
void compute_mean_variance(const float* input, float* mean,
                         float* variance, int batch_size,
                         int hidden_size, float eps);
void normalize_and_scale(const float* input, const float* mean,
                        const float* variance, const float* weight,
                        const float* bias, float* output,
                        int batch_size, int hidden_size);

#endif 