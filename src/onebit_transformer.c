/**
 * @file onebit_transformer.c
 * @brief Implementation of transformer architecture with binary optimization
 */

#include "onebit/onebit_transformer.h"
#include "onebit/onebit_ops.h"
#include "onebit/onebit_memory.h"
#include "onebit/onebit_error.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// KV Cache structure for efficient inference
typedef struct {
    // Key cache
    void* key_cache;
    // Value cache
    void* value_cache;
    // Current position in the cache
    size_t current_len;
    // Maximum sequence length
    size_t max_seq_len;
    // Whether the cache is initialized
    bool initialized;
} KVCache;

// Initialize KV cache for inference
static int kv_cache_init(OneBitContext* ctx, KVCache* cache, 
                        size_t batch_size, size_t num_heads, 
                        size_t head_dim, size_t max_seq_len) {
    if (!ctx || !cache) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    size_t cache_size = batch_size * num_heads * max_seq_len * head_dim * sizeof(float);
    
    cache->key_cache = onebit_malloc(ctx, cache_size);
    if (!cache->key_cache) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    cache->value_cache = onebit_malloc(ctx, cache_size);
    if (!cache->value_cache) {
        onebit_free(ctx, cache->key_cache);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Initialize cache
    memset(cache->key_cache, 0, cache_size);
    memset(cache->value_cache, 0, cache_size);
    
    cache->current_len = 0;
    cache->max_seq_len = max_seq_len;
    cache->initialized = true;
    
    return ONEBIT_SUCCESS;
}

// Free KV cache
static void kv_cache_free(OneBitContext* ctx, KVCache* cache) {
    if (!ctx || !cache) {
        return;
    }
    
    if (cache->key_cache) {
        onebit_free(ctx, cache->key_cache);
        cache->key_cache = NULL;
    }
    
    if (cache->value_cache) {
        onebit_free(ctx, cache->value_cache);
        cache->value_cache = NULL;
    }
    
    cache->current_len = 0;
    cache->max_seq_len = 0;
    cache->initialized = false;
}

// Initialize attention layer
int attention_init(OneBitContext* ctx, AttentionLayer* layer, const AttentionConfig* config) {
    if (!ctx || !layer || !config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Copy configuration
    memcpy(&layer->config, config, sizeof(AttentionConfig));
    layer->is_binary = config->use_binary;
    
    // Calculate dimensions
    size_t hidden_size = config->hidden_size;
    size_t num_heads = config->num_heads;
    size_t head_dim = hidden_size / num_heads;
    
    // Validate dimensions
    if (hidden_size % num_heads != 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Allocate weights and biases
    if (config->use_binary) {
        // Create binary tensors for query, key, value projections
        int64_t qkv_shape[2] = {hidden_size, hidden_size};
        
        // Initialize binary weight tensors
        int result = binary_tensor_init(ctx, &layer->weights.binary_weights.query_weights, 
                                     qkv_shape, 2, false);
        if (result != ONEBIT_SUCCESS) {
            return result;
        }
        
        result = binary_tensor_init(ctx, &layer->weights.binary_weights.key_weights, 
                                 qkv_shape, 2, false);
        if (result != ONEBIT_SUCCESS) {
            binary_tensor_free(ctx, &layer->weights.binary_weights.query_weights);
            return result;
        }
        
        result = binary_tensor_init(ctx, &layer->weights.binary_weights.value_weights, 
                                 qkv_shape, 2, false);
        if (result != ONEBIT_SUCCESS) {
            binary_tensor_free(ctx, &layer->weights.binary_weights.query_weights);
            binary_tensor_free(ctx, &layer->weights.binary_weights.key_weights);
            return result;
        }
        
        result = binary_tensor_init(ctx, &layer->weights.binary_weights.output_weights, 
                                 qkv_shape, 2, false);
        if (result != ONEBIT_SUCCESS) {
            binary_tensor_free(ctx, &layer->weights.binary_weights.query_weights);
            binary_tensor_free(ctx, &layer->weights.binary_weights.key_weights);
            binary_tensor_free(ctx, &layer->weights.binary_weights.value_weights);
            return result;
        }
    } else {
        // Allocate regular float weights
        size_t weight_size = hidden_size * hidden_size * sizeof(float);
        
        layer->weights.float_weights.query_weights = (float*)onebit_malloc(ctx, weight_size);
        if (!layer->weights.float_weights.query_weights) {
            return ONEBIT_ERROR_MEMORY;
        }
        
        layer->weights.float_weights.key_weights = (float*)onebit_malloc(ctx, weight_size);
        if (!layer->weights.float_weights.key_weights) {
            onebit_free(ctx, layer->weights.float_weights.query_weights);
            return ONEBIT_ERROR_MEMORY;
        }
        
        layer->weights.float_weights.value_weights = (float*)onebit_malloc(ctx, weight_size);
        if (!layer->weights.float_weights.value_weights) {
            onebit_free(ctx, layer->weights.float_weights.query_weights);
            onebit_free(ctx, layer->weights.float_weights.key_weights);
            return ONEBIT_ERROR_MEMORY;
        }
        
        layer->weights.float_weights.output_weights = (float*)onebit_malloc(ctx, weight_size);
        if (!layer->weights.float_weights.output_weights) {
            onebit_free(ctx, layer->weights.float_weights.query_weights);
            onebit_free(ctx, layer->weights.float_weights.key_weights);
            onebit_free(ctx, layer->weights.float_weights.value_weights);
            return ONEBIT_ERROR_MEMORY;
        }
        
        // Initialize weights to small random values
        for (size_t i = 0; i < hidden_size * hidden_size; i++) {
            float rand_val = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
            layer->weights.float_weights.query_weights[i] = rand_val;
            layer->weights.float_weights.key_weights[i] = rand_val;
            layer->weights.float_weights.value_weights[i] = rand_val;
            layer->weights.float_weights.output_weights[i] = rand_val;
        }
    }
    
    // Allocate biases
    size_t bias_size = hidden_size * sizeof(float);
    
    layer->query_bias = (float*)onebit_malloc(ctx, bias_size);
    if (!layer->query_bias) {
        attention_cleanup(ctx, layer);
        return ONEBIT_ERROR_MEMORY;
    }
    
    layer->key_bias = (float*)onebit_malloc(ctx, bias_size);
    if (!layer->key_bias) {
        attention_cleanup(ctx, layer);
        return ONEBIT_ERROR_MEMORY;
    }
    
    layer->value_bias = (float*)onebit_malloc(ctx, bias_size);
    if (!layer->value_bias) {
        attention_cleanup(ctx, layer);
        return ONEBIT_ERROR_MEMORY;
    }
    
    layer->output_bias = (float*)onebit_malloc(ctx, bias_size);
    if (!layer->output_bias) {
        attention_cleanup(ctx, layer);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Initialize biases to zero
    memset(layer->query_bias, 0, bias_size);
    memset(layer->key_bias, 0, bias_size);
    memset(layer->value_bias, 0, bias_size);
    memset(layer->output_bias, 0, bias_size);
    
    return ONEBIT_SUCCESS;
}

// Free attention layer resources
void attention_cleanup(OneBitContext* ctx, AttentionLayer* layer) {
    if (!ctx || !layer) {
        return;
    }
    
    if (layer->is_binary) {
        binary_tensor_free(ctx, &layer->weights.binary_weights.query_weights);
        binary_tensor_free(ctx, &layer->weights.binary_weights.key_weights);
        binary_tensor_free(ctx, &layer->weights.binary_weights.value_weights);
        binary_tensor_free(ctx, &layer->weights.binary_weights.output_weights);
    } else {
        if (layer->weights.float_weights.query_weights) {
            onebit_free(ctx, layer->weights.float_weights.query_weights);
        }
        if (layer->weights.float_weights.key_weights) {
            onebit_free(ctx, layer->weights.float_weights.key_weights);
        }
        if (layer->weights.float_weights.value_weights) {
            onebit_free(ctx, layer->weights.float_weights.value_weights);
        }
        if (layer->weights.float_weights.output_weights) {
            onebit_free(ctx, layer->weights.float_weights.output_weights);
        }
    }
    
    if (layer->query_bias) onebit_free(ctx, layer->query_bias);
    if (layer->key_bias) onebit_free(ctx, layer->key_bias);
    if (layer->value_bias) onebit_free(ctx, layer->value_bias);
    if (layer->output_bias) onebit_free(ctx, layer->output_bias);
    
    // Reset pointers
    memset(layer, 0, sizeof(AttentionLayer));
}

// Apply scaled dot-product attention
static int scaled_dot_product_attention(OneBitContext* ctx,
                                      const float* query, const float* key, const float* value,
                                      size_t batch_size, size_t num_heads, size_t seq_len,
                                      size_t head_dim, bool causal_mask, float* output) {
    if (!ctx || !query || !key || !value || !output) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    const float scale = 1.0f / sqrtf((float)head_dim);
    size_t attention_size = batch_size * num_heads * seq_len * seq_len * sizeof(float);
    
    // Allocate attention scores matrix
    float* attention_scores = (float*)onebit_malloc(ctx, attention_size);
    if (!attention_scores) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Compute Q*K^T for each batch and head
    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            for (size_t i = 0; i < seq_len; i++) {
                for (size_t j = 0; j < seq_len; j++) {
                    // Skip if using causal mask and j > i
                    if (causal_mask && j > i) {
                        // Set to large negative value
                        attention_scores[(b * num_heads + h) * seq_len * seq_len + i * seq_len + j] = -1e9f;
                        continue;
                    }
                    
                    // Compute dot product
                    float dot = 0.0f;
                    for (size_t k = 0; k < head_dim; k++) {
                        size_t q_idx = ((b * num_heads + h) * seq_len + i) * head_dim + k;
                        size_t k_idx = ((b * num_heads + h) * seq_len + j) * head_dim + k;
                        dot += query[q_idx] * key[k_idx];
                    }
                    
                    // Apply scaling
                    attention_scores[(b * num_heads + h) * seq_len * seq_len + i * seq_len + j] = dot * scale;
                }
                
                // Apply softmax to each row
                float max_val = -INFINITY;
                size_t row_offset = (b * num_heads + h) * seq_len * seq_len + i * seq_len;
                
                // Find max for numerical stability
                for (size_t j = 0; j < seq_len; j++) {
                    if (attention_scores[row_offset + j] > max_val) {
                        max_val = attention_scores[row_offset + j];
                    }
                }
                
                // Compute exp and sum
                float sum_exp = 0.0f;
                for (size_t j = 0; j < seq_len; j++) {
                    attention_scores[row_offset + j] = expf(attention_scores[row_offset + j] - max_val);
                    sum_exp += attention_scores[row_offset + j];
                }
                
                // Normalize
                for (size_t j = 0; j < seq_len; j++) {
                    attention_scores[row_offset + j] /= sum_exp;
                }
            }
        }
    }
    
    // Compute attention_scores * V
    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            for (size_t i = 0; i < seq_len; i++) {
                for (size_t k = 0; k < head_dim; k++) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < seq_len; j++) {
                        size_t attn_idx = (b * num_heads + h) * seq_len * seq_len + i * seq_len + j;
                        size_t v_idx = ((b * num_heads + h) * seq_len + j) * head_dim + k;
                        sum += attention_scores[attn_idx] * value[v_idx];
                    }
                    size_t out_idx = ((b * num_heads + h) * seq_len + i) * head_dim + k;
                    output[out_idx] = sum;
                }
            }
        }
    }
    
    // Free temporary memory
    onebit_free(ctx, attention_scores);
    
    return ONEBIT_SUCCESS;
}

// Forward pass through attention layer
int attention_forward(OneBitContext* ctx, AttentionLayer* layer, 
                     const Tensor* input, Tensor* output,
                     const Tensor* mask, void* kv_cache_ptr) {
    if (!ctx || !layer || !input || !output) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Extract dimensions
    size_t batch_size = input->shape[0];
    size_t seq_len = input->shape[1];
    size_t hidden_size = layer->config.hidden_size;
    size_t num_heads = layer->config.num_heads;
    size_t head_dim = hidden_size / num_heads;
    
    // Validate input dimensions
    if (input->ndim != 3 || input->shape[2] != hidden_size) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Ensure output has correct shape
    if (output->ndim != 3 || 
        output->shape[0] != batch_size || 
        output->shape[1] != seq_len || 
        output->shape[2] != hidden_size) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Allocate intermediate buffers
    float* query = (float*)onebit_malloc(ctx, batch_size * seq_len * hidden_size * sizeof(float));
    float* key = (float*)onebit_malloc(ctx, batch_size * seq_len * hidden_size * sizeof(float));
    float* value = (float*)onebit_malloc(ctx, batch_size * seq_len * hidden_size * sizeof(float));
    float* attention_output = (float*)onebit_malloc(ctx, batch_size * seq_len * hidden_size * sizeof(float));
    
    if (!query || !key || !value || !attention_output) {
        if (query) onebit_free(ctx, query);
        if (key) onebit_free(ctx, key);
        if (value) onebit_free(ctx, value);
        if (attention_output) onebit_free(ctx, attention_output);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Project input to query, key, value
    int result;
    const float* input_data = (const float*)input->data;
    
    if (layer->is_binary) {
        // Binary weights projection
        int64_t proj_dims[3] = {batch_size * seq_len, hidden_size, hidden_size};
        
        // Project query
        result = compute_binary_matmul(
            layer->weights.binary_weights.query_weights.data,
            input_data,
            query,
            proj_dims,
            layer->weights.binary_weights.query_weights.scales
        );
        if (result != ONEBIT_SUCCESS) goto cleanup;
        
        // Project key
        result = compute_binary_matmul(
            layer->weights.binary_weights.key_weights.data,
            input_data,
            key,
            proj_dims,
            layer->weights.binary_weights.key_weights.scales
        );
        if (result != ONEBIT_SUCCESS) goto cleanup;
        
        // Project value
        result = compute_binary_matmul(
            layer->weights.binary_weights.value_weights.data,
            input_data,
            value,
            proj_dims,
            layer->weights.binary_weights.value_weights.scales
        );
        if (result != ONEBIT_SUCCESS) goto cleanup;
    } else {
        // Float weights projection
        int64_t matmul_dims[3] = {batch_size * seq_len, hidden_size, hidden_size};
        
        // Project query
        result = compute_matmul(
            input_data,
            layer->weights.float_weights.query_weights,
            query,
            matmul_dims
        );
        if (result != ONEBIT_SUCCESS) goto cleanup;
        
        // Project key
        result = compute_matmul(
            input_data,
            layer->weights.float_weights.key_weights,
            key,
            matmul_dims
        );
        if (result != ONEBIT_SUCCESS) goto cleanup;
        
        // Project value
        result = compute_matmul(
            input_data,
            layer->weights.float_weights.value_weights,
            value,
            matmul_dims
        );
        if (result != ONEBIT_SUCCESS) goto cleanup;
    }
    
    // Add biases
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size * seq_len; i++) {
        for (size_t j = 0; j < hidden_size; j++) {
            query[i * hidden_size + j] += layer->query_bias[j];
            key[i * hidden_size + j] += layer->key_bias[j];
            value[i * hidden_size + j] += layer->value_bias[j];
        }
    }
    
    // Apply self-attention
    result = scaled_dot_product_attention(
        ctx,
        query, key, value,
        batch_size, num_heads, seq_len, head_dim,
        layer->config.causal_mask,
        attention_output
    );
    if (result != ONEBIT_SUCCESS) goto cleanup;
    
    // Project attention output back to hidden size
    if (layer->is_binary) {
        int64_t proj_dims[3] = {batch_size * seq_len, hidden_size, hidden_size};
        
        result = compute_binary_matmul(
            layer->weights.binary_weights.output_weights.data,
            attention_output,
            output->data,
            proj_dims,
            layer->weights.binary_weights.output_weights.scales
        );
    } else {
        int64_t matmul_dims[3] = {batch_size * seq_len, hidden_size, hidden_size};
        
        result = compute_matmul(
            attention_output,
            layer->weights.float_weights.output_weights,
            output->data,
            matmul_dims
        );
    }
    if (result != ONEBIT_SUCCESS) goto cleanup;
    
    // Add output bias
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size * seq_len; i++) {
        for (size_t j = 0; j < hidden_size; j++) {
            ((float*)output->data)[i * hidden_size + j] += layer->output_bias[j];
        }
    }
    
cleanup:
    onebit_free(ctx, query);
    onebit_free(ctx, key);
    onebit_free(ctx, value);
    onebit_free(ctx, attention_output);
    
    return result;
}

// Initialize a feed-forward layer
int ffn_init(OneBitContext* ctx, FFNLayer* layer, const FFNConfig* config) {
    if (!ctx || !layer || !config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Copy configuration
    memcpy(&layer->config, config, sizeof(FFNConfig));
    layer->is_binary = config->use_binary;
    
    // Calculate dimensions
    size_t hidden_size = config->hidden_size;
    size_t intermediate_size = config->intermediate_size;
    
    if (config->use_binary) {
        // Create binary tensors
        int64_t intermediate_shape[2] = {intermediate_size, hidden_size};
        int64_t output_shape[2] = {hidden_size, intermediate_size};
        
        // Initialize binary tensors
        int result = binary_tensor_init(ctx, &layer->weights.binary_weights.intermediate_weights,
                                     intermediate_shape, 2, false);
        if (result != ONEBIT_SUCCESS) {
            return result;
        }
        
        result = binary_tensor_init(ctx, &layer->weights.binary_weights.output_weights,
                                 output_shape, 2, false);
        if (result != ONEBIT_SUCCESS) {
            binary_tensor_free(ctx, &layer->weights.binary_weights.intermediate_weights);
            return result;
        }
    } else {
        // Allocate float tensors
        size_t intermediate_size_bytes = hidden_size * intermediate_size * sizeof(float);
        size_t output_size_bytes = intermediate_size * hidden_size * sizeof(float);
        
        layer->weights.float_weights.intermediate_weights = (float*)onebit_malloc(ctx, intermediate_size_bytes);
        if (!layer->weights.float_weights.intermediate_weights) {
            return ONEBIT_ERROR_MEMORY;
        }
        
        layer->weights.float_weights.output_weights = (float*)onebit_malloc(ctx, output_size_bytes);
        if (!layer->weights.float_weights.output_weights) {
            onebit_free(ctx, layer->weights.float_weights.intermediate_weights);
            return ONEBIT_ERROR_MEMORY;
        }
        
        // Initialize with small random values
        for (size_t i = 0; i < hidden_size * intermediate_size; i++) {
            layer->weights.float_weights.intermediate_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        }
        
        for (size_t i = 0; i < intermediate_size * hidden_size; i++) {
            layer->weights.float_weights.output_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        }
    }
    
    // Allocate biases
    layer->intermediate_bias = (float*)onebit_malloc(ctx, intermediate_size * sizeof(float));
    if (!layer->intermediate_bias) {
        ffn_cleanup(ctx, layer);
        return ONEBIT_ERROR_MEMORY;
    }
    
    layer->output_bias = (float*)onebit_malloc(ctx, hidden_size * sizeof(float));
    if (!layer->output_bias) {
        ffn_cleanup(ctx, layer);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Initialize biases to zero
    memset(layer->intermediate_bias, 0, intermediate_size * sizeof(float));
    memset(layer->output_bias, 0, hidden_size * sizeof(float));
    
    return ONEBIT_SUCCESS;
}

// Free feed-forward layer resources
void ffn_cleanup(OneBitContext* ctx, FFNLayer* layer) {
    if (!ctx || !layer) {
        return;
    }
    
    if (layer->is_binary) {
        binary_tensor_free(ctx, &layer->weights.binary_weights.intermediate_weights);
        binary_tensor_free(ctx, &layer->weights.binary_weights.output_weights);
    } else {
        if (layer->weights.float_weights.intermediate_weights) {
            onebit_free(ctx, layer->weights.float_weights.intermediate_weights);
        }
        if (layer->weights.float_weights.output_weights) {
            onebit_free(ctx, layer->weights.float_weights.output_weights);
        }
    }
    
    if (layer->intermediate_bias) {
        onebit_free(ctx, layer->intermediate_bias);
    }
    if (layer->output_bias) {
        onebit_free(ctx, layer->output_bias);
    }
    
    // Reset pointers
    memset(layer, 0, sizeof(FFNLayer));
}

// Forward pass through feed-forward layer
int ffn_forward(OneBitContext* ctx, FFNLayer* layer,
               const Tensor* input, Tensor* output) {
    if (!ctx || !layer || !input || !output) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Extract dimensions
    size_t batch_size = input->shape[0];
    size_t seq_len = input->shape[1];
    size_t hidden_size = layer->config.hidden_size;
    size_t intermediate_size = layer->config.intermediate_size;
    
    // Validate input dimensions
    if (input->ndim != 3 || input->shape[2] != hidden_size) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Validate output dimensions
    if (output->ndim != 3 || 
        output->shape[0] != batch_size || 
        output->shape[1] != seq_len || 
        output->shape[2] != hidden_size) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Allocate intermediate tensor
    float* intermediate = (float*)onebit_malloc(ctx, batch_size * seq_len * intermediate_size * sizeof(float));
    if (!intermediate) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    int result;
    const float* input_data = (const float*)input->data;
    
    // First projection: input -> intermediate
    if (layer->is_binary) {
        int64_t proj_dims[3] = {batch_size * seq_len, intermediate_size, hidden_size};
        
        result = compute_binary_matmul(
            layer->weights.binary_weights.intermediate_weights.data,
            input_data,
            intermediate,
            proj_dims,
            layer->weights.binary_weights.intermediate_weights.scales
        );
    } else {
        int64_t matmul_dims[3] = {batch_size * seq_len, intermediate_size, hidden_size};
        
        result = compute_matmul(
            input_data,
            layer->weights.float_weights.intermediate_weights,
            intermediate,
            matmul_dims
        );
    }
    if (result != ONEBIT_SUCCESS) {
        onebit_free(ctx, intermediate);
        return result;
    }
    
    // Add intermediate bias
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size * seq_len; i++) {
        for (size_t j = 0; j < intermediate_size; j++) {
            intermediate[i * intermediate_size + j] += layer->intermediate_bias[j];
        }
    }
    
    // Apply GELU activation
    result = compute_gelu(intermediate, intermediate, batch_size * seq_len * intermediate_size);
    if (result != ONEBIT_SUCCESS) {
        onebit_free(ctx, intermediate);
        return result;
    }
    
    // Second projection: intermediate -> output
    if (layer->is_binary) {
        int64_t proj_dims[3] = {batch_size * seq_len, hidden_size, intermediate_size};
        
        result = compute_binary_matmul(
            layer->weights.binary_weights.output_weights.data,
            intermediate,
            output->data,
            proj_dims,
            layer->weights.binary_weights.output_weights.scales
        );
    } else {
        int64_t matmul_dims[3] = {batch_size * seq_len, hidden_size, intermediate_size};
        
        result = compute_matmul(
            intermediate,
            layer->weights.float_weights.output_weights,
            output->data,
            matmul_dims
        );
    }
    if (result != ONEBIT_SUCCESS) {
        onebit_free(ctx, intermediate);
        return result;
    }
    
    // Add output bias
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size * seq_len; i++) {
        for (size_t j = 0; j < hidden_size; j++) {
            ((float*)output->data)[i * hidden_size + j] += layer->output_bias[j];
        }
    }
    
    onebit_free(ctx, intermediate);
    return ONEBIT_SUCCESS;
}

// Initialize a transformer block
int transformer_block_init(OneBitContext* ctx, TransformerBlock* block,
                         size_t hidden_size, size_t num_heads,
                         size_t ff_size, bool use_binary) {
    if (!ctx || !block) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    block->is_binary = use_binary;
    block->dropout_prob = 0.1f; // Default dropout probability
    
    // Initialize attention layer
    AttentionConfig attn_config = {
        .hidden_size = hidden_size,
        .num_heads = num_heads,
        .dropout_prob = 0.1f,
        .use_binary = use_binary,
        .causal_mask = true
    };
    
    int result = attention_init(ctx, &block->attention, &attn_config);
    if (result != ONEBIT_SUCCESS) {
        return result;
    }
    
    // Initialize feed-forward layer
    FFNConfig ffn_config = {
        .hidden_size = hidden_size,
        .intermediate_size = ff_size,
        .dropout_prob = 0.1f,
        .use_binary = use_binary
    };
    
    result = ffn_init(ctx, &block->ffn, &ffn_config);
    if (result != ONEBIT_SUCCESS) {
        attention_cleanup(ctx, &block->attention);
        return result;
    }
    
    // Allocate layer normalization parameters
    block->attention_ln_weight = (float*)onebit_malloc(ctx, hidden_size * sizeof(float));
    block->attention_ln_bias = (float*)onebit_malloc(ctx, hidden_size * sizeof(float));
    block->ffn_ln_weight = (float*)onebit_malloc(ctx, hidden_size * sizeof(float));
    block->ffn_ln_bias = (float*)onebit_malloc(ctx, hidden_size * sizeof(float));
    
    if (!block->attention_ln_weight || !block->attention_ln_bias ||
        !block->ffn_ln_weight || !block->ffn_ln_bias) {
        transformer_block_cleanup(ctx, block);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Initialize layer norm parameters
    for (size_t i = 0; i < hidden_size; i++) {
        block->attention_ln_weight[i] = 1.0f;
        block->attention_ln_bias[i] = 0.0f;
        block->ffn_ln_weight[i] = 1.0f;
        block->ffn_ln_bias[i] = 0.0f;
    }
    
    return ONEBIT_SUCCESS;
}

// Clean up a transformer block
void transformer_block_cleanup(OneBitContext* ctx, TransformerBlock* block) {
    if (!ctx || !block) {
        return;
    }
    
    // Clean up attention and ffn layers
    attention_cleanup(ctx, &block->attention);
    ffn_cleanup(ctx, &block->ffn);
    
    // Free layer norm parameters
    if (block->attention_ln_weight) onebit_free(ctx, block->attention_ln_weight);
    if (block->attention_ln_bias) onebit_free(ctx, block->attention_ln_bias);
    if (block->ffn_ln_weight) onebit_free(ctx, block->ffn_ln_weight);
    if (block->ffn_ln_bias) onebit_free(ctx, block->ffn_ln_bias);
    
    // Reset pointers
    block->attention_ln_weight = NULL;
    block->attention_ln_bias = NULL;
    block->ffn_ln_weight = NULL;
    block->ffn_ln_bias = NULL;
}

// Apply layer normalization
static int layer_norm(OneBitContext* ctx, const float* input, 
                    float* output, size_t batch_size, size_t seq_len,
                    size_t hidden_size, const float* weight, const float* bias) {
    if (!ctx || !input || !output || !weight || !bias) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Layer norm for each token position in batch
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size * seq_len; i++) {
        // Compute mean
        float mean = 0.0f;
        for (size_t j = 0; j < hidden_size; j++) {
            mean += input[i * hidden_size + j];
        }
        mean /= hidden_size;
        
        // Compute variance
        float var = 0.0f;
        for (size_t j = 0; j < hidden_size; j++) {
            float diff = input[i * hidden_size + j] - mean;
            var += diff * diff;
        }
        var /= hidden_size;
        
        // Normalize, scale and shift
        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (size_t j = 0; j < hidden_size; j++) {
            output[i * hidden_size + j] = 
                (input[i * hidden_size + j] - mean) * inv_std * weight[j] + bias[j];
        }
    }
    
    return ONEBIT_SUCCESS;
}

// Forward pass through transformer block
int transformer_block_forward(OneBitContext* ctx, TransformerBlock* block,
                            const Tensor* input, Tensor* output,
                            const Tensor* mask, void* kv_cache) {
    if (!ctx || !block || !input || !output) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Extract dimensions
    size_t batch_size = input->shape[0];
    size_t seq_len = input->shape[1];
    size_t hidden_size = input->shape[2];
    
    // Validate output dimensions
    if (output->ndim != 3 || 
        output->shape[0] != batch_size || 
        output->shape[1] != seq_len || 
        output->shape[2] != hidden_size) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Allocate temporary buffers
    float* norm_output = (float*)onebit_malloc(ctx, batch_size * seq_len * hidden_size * sizeof(float));
    float* attn_output = (float*)onebit_malloc(ctx, batch_size * seq_len * hidden_size * sizeof(float));
    float* residual = (float*)onebit_malloc(ctx, batch_size * seq_len * hidden_size * sizeof(float));
    
    if (!norm_output || !attn_output || !residual) {
        if (norm_output) onebit_free(ctx, norm_output);
        if (attn_output) onebit_free(ctx, attn_output);
        if (residual) onebit_free(ctx, residual);
        return ONEBIT_ERROR_MEMORY;
    }
    
    int result;
    
    // Apply attention layer norm
    result = layer_norm(ctx, (const float*)input->data, norm_output, 
                      batch_size, seq_len, hidden_size,
                      block->attention_ln_weight, block->attention_ln_bias);
    if (result != ONEBIT_SUCCESS) goto cleanup;
    
    // Create tensor for normalized input
    Tensor norm_tensor = {
        .data = norm_output,
        .shape = input->shape,
        .ndim = input->ndim,
        .dtype = input->dtype
    };
    
    // Apply attention
    Tensor attn_tensor = {
        .data = attn_output,
        .shape = input->shape,
        .ndim = input->ndim,
        .dtype = input->dtype
    };
    
    result = attention_forward(ctx, &block->attention, &norm_tensor, &attn_tensor, mask, kv_cache);
    if (result != ONEBIT_SUCCESS) goto cleanup;
    
    // Add residual connection
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size * seq_len * hidden_size; i++) {
        residual[i] = attn_output[i] + ((const float*)input->data)[i];
    }
    
    // Apply ffn layer norm
    result = layer_norm(ctx, residual, norm_output, 
                      batch_size, seq_len, hidden_size,
                      block->ffn_ln_weight, block->ffn_ln_bias);
    if (result != ONEBIT_SUCCESS) goto cleanup;
    
    // Update norm tensor data
    norm_tensor.data = norm_output;
    
    // Apply feed-forward network
    Tensor output_tensor = {
        .data = output->data,
        .shape = output->shape,
        .ndim = output->ndim,
        .dtype = output->dtype
    };
    
    result = ffn_forward(ctx, &block->ffn, &norm_tensor, &output_tensor);
    if (result != ONEBIT_SUCCESS) goto cleanup;
    
    // Add second residual connection
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size * seq_len * hidden_size; i++) {
        ((float*)output->data)[i] += residual[i];
    }
    
cleanup:
    if (norm_output) onebit_free(ctx, norm_output);
    if (attn_output) onebit_free(ctx, attn_output);
    if (residual) onebit_free(ctx, residual);
    
    return result;
}

// Initialize a transformer model
int transformer_model_init(OneBitContext* ctx, TransformerModel* model,
                         size_t hidden_size, size_t num_layers,
                         size_t num_heads, size_t ff_size,
                         size_t vocab_size, size_t max_seq_len,
                         bool use_binary) {
    if (!ctx || !model) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Initialize model configuration
    model->hidden_size = hidden_size;
    model->vocab_size = vocab_size;
    model->max_seq_len = max_seq_len;
    model->is_binary = use_binary;
    model->num_blocks = num_layers;
    
    // Allocate token embeddings
    size_t embed_size = vocab_size * hidden_size * sizeof(float);
    model->token_embeddings = (float*)onebit_malloc(ctx, embed_size);
    if (!model->token_embeddings) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Allocate position embeddings
    size_t pos_embed_size = max_seq_len * hidden_size * sizeof(float);
    model->position_embeddings = (float*)onebit_malloc(ctx, pos_embed_size);
    if (!model->position_embeddings) {
        transformer_model_cleanup(ctx, model);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Initialize embeddings with small random values
    for (size_t i = 0; i < vocab_size * hidden_size; i++) {
        model->token_embeddings[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
    
    for (size_t i = 0; i < max_seq_len * hidden_size; i++) {
        model->position_embeddings[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
    
    // Allocate transformer blocks
    model->blocks = (TransformerBlock*)onebit_malloc(ctx, num_layers * sizeof(TransformerBlock));
    if (!model->blocks) {
        transformer_model_cleanup(ctx, model);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Initialize transformer blocks
    for (size_t i = 0; i < num_layers; i++) {
        int result = transformer_block_init(ctx, &model->blocks[i], 
                                          hidden_size, num_heads,
                                          ff_size, use_binary);
        if (result != ONEBIT_SUCCESS) {
            // Clean up previously initialized blocks
            for (size_t j = 0; j < i; j++) {
                transformer_block_cleanup(ctx, &model->blocks[j]);
            }
            transformer_model_cleanup(ctx, model);
            return result;
        }
    }
    
    // Allocate output layer
    if (use_binary) {
        int64_t output_shape[2] = {vocab_size, hidden_size};
        int result = binary_tensor_init(ctx, &model->output.output_weights, 
                                     output_shape, 2, false);
        if (result != ONEBIT_SUCCESS) {
            transformer_model_cleanup(ctx, model);
            return result;
        }
    } else {
        size_t output_size = vocab_size * hidden_size * sizeof(float);
        model->output.output_weights = (float*)onebit_malloc(ctx, output_size);
        if (!model->output.output_weights) {
            transformer_model_cleanup(ctx, model);
            return ONEBIT_ERROR_MEMORY;
        }
        
        // Initialize with small random values
        for (size_t i = 0; i < vocab_size * hidden_size; i++) {
            model->output.output_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
        }
    }
    
    // Allocate output bias
    model->output_bias = (float*)onebit_malloc(ctx, vocab_size * sizeof(float));
    if (!model->output_bias) {
        transformer_model_cleanup(ctx, model);
        return ONEBIT_ERROR_MEMORY;
    }
    memset(model->output_bias, 0, vocab_size * sizeof(float));
    
    // Allocate layer normalization parameters
    model->final_ln_weight = (float*)onebit_malloc(ctx, hidden_size * sizeof(float));
    model->final_ln_bias = (float*)onebit_malloc(ctx, hidden_size * sizeof(float));
    if (!model->final_ln_weight || !model->final_ln_bias) {
        transformer_model_cleanup(ctx, model);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Initialize layer norm parameters
    for (size_t i = 0; i < hidden_size; i++) {
        model->final_ln_weight[i] = 1.0f;
        model->final_ln_bias[i] = 0.0f;
    }
    
    return ONEBIT_SUCCESS;
}

// Cleanup a transformer model
void transformer_model_cleanup(OneBitContext* ctx, TransformerModel* model) {
    if (!ctx || !model) {
        return;
    }
    
    // Free embeddings
    if (model->token_embeddings) {
        onebit_free(ctx, model->token_embeddings);
        model->token_embeddings = NULL;
    }
    
    if (model->position_embeddings) {
        onebit_free(ctx, model->position_embeddings);
        model->position_embeddings = NULL;
    }
    
    // Free transformer blocks
    if (model->blocks) {
        for (size_t i = 0; i < model->num_blocks; i++) {
            transformer_block_cleanup(ctx, &model->blocks[i]);
        }
        onebit_free(ctx, model->blocks);
        model->blocks = NULL;
    }
    
    // Free output layer
    if (model->is_binary) {
        binary_tensor_free(ctx, &model->output.output_weights);
    } else if (model->output.output_weights) {
        onebit_free(ctx, model->output.output_weights);
        model->output.output_weights = NULL;
    }
    
    if (model->output_bias) {
        onebit_free(ctx, model->output_bias);
        model->output_bias = NULL;
    }
    
    // Free layer norm parameters
    if (model->final_ln_weight) {
        onebit_free(ctx, model->final_ln_weight);
        model->final_ln_weight = NULL;
    }
    
    if (model->final_ln_bias) {
        onebit_free(ctx, model->final_ln_bias);
        model->final_ln_bias = NULL;
    }
}

// Lookup embeddings for input tokens
static int lookup_embeddings(OneBitContext* ctx, const TransformerModel* model,
                           const int64_t* input_ids, size_t batch_size, size_t seq_len,
                           float* embeddings) {
    if (!ctx || !model || !input_ids || !embeddings) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    size_t hidden_size = model->hidden_size;
    
    // Process each token in the batch
    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            // Get token ID (clamp to valid range)
            int64_t token_id = input_ids[b * seq_len + s];
            if (token_id < 0) token_id = 0;
            if (token_id >= (int64_t)model->vocab_size) token_id = model->vocab_size - 1;
            
            // Get position (clamp to valid range)
            size_t pos = s;
            if (pos >= model->max_seq_len) pos = model->max_seq_len - 1;
            
            // Copy token embedding
            float* dest = embeddings + (b * seq_len + s) * hidden_size;
            const float* token_embed = model->token_embeddings + token_id * hidden_size;
            const float* pos_embed = model->position_embeddings + pos * hidden_size;
            
            // Add token and positional embeddings
            for (size_t i = 0; i < hidden_size; i++) {
                dest[i] = token_embed[i] + pos_embed[i];
            }
        }
    }
    
    return ONEBIT_SUCCESS;
}

// Forward pass through transformer model
int transformer_model_forward(OneBitContext* ctx, TransformerModel* model,
                            const int64_t* input_ids, Tensor* output,
                            void* kv_cache_ptr) {
    if (!ctx || !model || !input_ids || !output) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Extract dimensions
    size_t batch_size = output->shape[0];
    size_t seq_len = output->shape[1];
    size_t hidden_size = model->hidden_size;
    size_t vocab_size = model->vocab_size;
    
    // Validate output tensor dimensions
    if (output->ndim != 3 || output->shape[2] != vocab_size) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Allocate temporary buffers
    float* embeddings = (float*)onebit_malloc(ctx, batch_size * seq_len * hidden_size * sizeof(float));
    float* hidden_states = (float*)onebit_malloc(ctx, batch_size * seq_len * hidden_size * sizeof(float));
    float* layer_output = (float*)onebit_malloc(ctx, batch_size * seq_len * hidden_size * sizeof(float));
    float* final_hidden = (float*)onebit_malloc(ctx, batch_size * seq_len * hidden_size * sizeof(float));
    
    if (!embeddings || !hidden_states || !layer_output || !final_hidden) {
        if (embeddings) onebit_free(ctx, embeddings);
        if (hidden_states) onebit_free(ctx, hidden_states);
        if (layer_output) onebit_free(ctx, layer_output);
        if (final_hidden) onebit_free(ctx, final_hidden);
        return ONEBIT_ERROR_MEMORY;
    }
    
    int result;
    
    // Look up embeddings
    result = lookup_embeddings(ctx, model, input_ids, batch_size, seq_len, embeddings);
    if (result != ONEBIT_SUCCESS) goto cleanup;
    
    // Copy embeddings to hidden states
    memcpy(hidden_states, embeddings, batch_size * seq_len * hidden_size * sizeof(float));
    
    // Create tensors for transformer blocks
    Tensor hidden_tensor = {
        .data = hidden_states,
        .shape = {batch_size, seq_len, hidden_size},
        .ndim = 3,
        .dtype = TENSOR_TYPE_FLOAT32
    };
    
    Tensor layer_tensor = {
        .data = layer_output,
        .shape = {batch_size, seq_len, hidden_size},
        .ndim = 3,
        .dtype = TENSOR_TYPE_FLOAT32
    };
    
    // Process each transformer block
    for (size_t i = 0; i < model->num_blocks; i++) {
        // Set up KV cache for this layer if provided
        void* layer_kv_cache = NULL;
        if (kv_cache_ptr) {
            KVCache* kv_array = (KVCache*)kv_cache_ptr;
            layer_kv_cache = &kv_array[i];
        }
        
        // Forward pass through transformer block
        result = transformer_block_forward(ctx, &model->blocks[i], 
                                         &hidden_tensor, &layer_tensor,
                                         NULL, layer_kv_cache);
        if (result != ONEBIT_SUCCESS) goto cleanup;
        
        // Swap buffers for next layer
        float* temp = hidden_states;
        hidden_states = layer_output;
        layer_output = temp;
        
        // Update tensor pointers
        hidden_tensor.data = hidden_states;
        layer_tensor.data = layer_output;
    }
    
    // Apply final layer norm
    result = layer_norm(ctx, hidden_states, final_hidden, 
                      batch_size, seq_len, hidden_size,
                      model->final_ln_weight, model->final_ln_bias);
    if (result != ONEBIT_SUCCESS) goto cleanup;
    
    // Project to vocabulary (output logits)
    if (model->is_binary) {
        int64_t proj_dims[3] = {batch_size * seq_len, vocab_size, hidden_size};
        
        result = compute_binary_matmul(
            model->output.output_weights.data,
            final_hidden,
            output->data,
            proj_dims,
            model->output.output_weights.scales
        );
    } else {
        int64_t matmul_dims[3] = {batch_size * seq_len, vocab_size, hidden_size};
        
        result = compute_matmul(
            final_hidden,
            model->output.output_weights,
            output->data,
            matmul_dims
        );
    }
    if (result != ONEBIT_SUCCESS) goto cleanup;
    
    // Add output bias
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size * seq_len; i++) {
        for (size_t j = 0; j < vocab_size; j++) {
            ((float*)output->data)[i * vocab_size + j] += model->output_bias[j];
        }
    }
    
cleanup:
    if (embeddings) onebit_free(ctx, embeddings);
    if (hidden_states) onebit_free(ctx, hidden_states);
    if (layer_output) onebit_free(ctx, layer_output);
    if (final_hidden) onebit_free(ctx, final_hidden);
    
    return result;
}

// Create KV cache for efficient inference
int transformer_create_kv_cache(OneBitContext* ctx, TransformerModel* model,
                              size_t batch_size, void** kv_cache) {
    if (!ctx || !model || !kv_cache) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Allocate array of KVCache structures (one per layer)
    KVCache* cache_array = (KVCache*)onebit_malloc(ctx, model->num_blocks * sizeof(KVCache));
    if (!cache_array) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Initialize each cache
    size_t num_heads = model->blocks[0].attention.config.num_heads;
    size_t head_dim = model->hidden_size / num_heads;
    size_t max_seq_len = model->max_seq_len;
    
    for (size_t i = 0; i < model->num_blocks; i++) {
        int result = kv_cache_init(ctx, &cache_array[i], 
                                batch_size, num_heads, head_dim, max_seq_len);
        if (result != ONEBIT_SUCCESS) {
            // Clean up previously initialized caches
            for (size_t j = 0; j < i; j++) {
                kv_cache_free(ctx, &cache_array[j]);
            }
            onebit_free(ctx, cache_array);
            return result;
        }
    }
    
    *kv_cache = cache_array;
    return ONEBIT_SUCCESS;
}

// Free KV cache
void transformer_free_kv_cache(OneBitContext* ctx, TransformerModel* model, void* kv_cache) {
    if (!ctx || !model || !kv_cache) {
        return;
    }
    
    KVCache* cache_array = (KVCache*)kv_cache;
    
    // Free each cache
    for (size_t i = 0; i < model->num_blocks; i++) {
        kv_cache_free(ctx, &cache_array[i]);
    }
    
    // Free the array
    onebit_free(ctx, cache_array);
} 