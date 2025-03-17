/**
 * @file onebit_kv_cache.c
 * @brief Key-Value cache implementation for efficient incremental decoding
 */

#include "onebit/onebit_kv_cache.h"
#include "onebit/onebit_errors.h"
#include <string.h>

int kv_cache_init(OneBitContext* ctx, KVCache* cache,
                size_t batch_size, size_t num_heads,
                size_t head_dim, size_t max_seq_len) {
    if (!ctx || !cache) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Initialize cache parameters
    cache->batch_size = batch_size;
    cache->num_heads = num_heads;
    cache->head_dim = head_dim;
    cache->max_seq_len = max_seq_len;
    cache->current_seq_len = 0;
    
    // Calculate cache size
    size_t cache_size = batch_size * num_heads * max_seq_len * head_dim;
    size_t cache_bytes = cache_size * sizeof(float);
    
    // Allocate key and value caches
    cache->key_cache = (float*)onebit_malloc(ctx, cache_bytes);
    cache->value_cache = (float*)onebit_malloc(ctx, cache_bytes);
    
    if (!cache->key_cache || !cache->value_cache) {
        if (cache->key_cache) {
            onebit_free(ctx, cache->key_cache);
            cache->key_cache = NULL;
        }
        if (cache->value_cache) {
            onebit_free(ctx, cache->value_cache);
            cache->value_cache = NULL;
        }
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Initialize cache data to zeros
    memset(cache->key_cache, 0, cache_bytes);
    memset(cache->value_cache, 0, cache_bytes);
    
    return ONEBIT_SUCCESS;
}

void kv_cache_free(OneBitContext* ctx, KVCache* cache) {
    if (!ctx || !cache) {
        return;
    }
    
    // Free cache buffers
    if (cache->key_cache) {
        onebit_free(ctx, cache->key_cache);
        cache->key_cache = NULL;
    }
    
    if (cache->value_cache) {
        onebit_free(ctx, cache->value_cache);
        cache->value_cache = NULL;
    }
    
    // Reset parameters
    cache->current_seq_len = 0;
}

int kv_cache_append(OneBitContext* ctx, KVCache* cache,
                  const float* new_keys, const float* new_values,
                  size_t new_seq_len) {
    if (!ctx || !cache || !new_keys || !new_values) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Check if appending would exceed max sequence length
    if (cache->current_seq_len + new_seq_len > cache->max_seq_len) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    size_t batch_size = cache->batch_size;
    size_t num_heads = cache->num_heads;
    size_t head_dim = cache->head_dim;
    size_t curr_len = cache->current_seq_len;
    
    // For each batch and head, copy new keys and values into the cache
    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            // Calculate offsets
            size_t cache_offset = (b * num_heads * cache->max_seq_len * head_dim) + 
                                 (h * cache->max_seq_len * head_dim) + 
                                 (curr_len * head_dim);
            
            size_t new_offset = (b * num_heads * new_seq_len * head_dim) + 
                               (h * new_seq_len * head_dim);
            
            // Copy new keys and values
            size_t copy_size = new_seq_len * head_dim * sizeof(float);
            memcpy(cache->key_cache + cache_offset, new_keys + new_offset, copy_size);
            memcpy(cache->value_cache + cache_offset, new_values + new_offset, copy_size);
        }
    }
    
    // Update current sequence length
    cache->current_seq_len += new_seq_len;
    
    return ONEBIT_SUCCESS;
}

int kv_cache_reset(OneBitContext* ctx, KVCache* cache) {
    if (!ctx || !cache) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Reset sequence length
    cache->current_seq_len = 0;
    
    return ONEBIT_SUCCESS;
}

int kv_cache_get_keys(const KVCache* cache, float** out_keys, size_t* out_seq_len) {
    if (!cache || !out_keys || !out_seq_len) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Return pointers and current sequence length
    *out_keys = cache->key_cache;
    *out_seq_len = cache->current_seq_len;
    
    return ONEBIT_SUCCESS;
}

int kv_cache_get_values(const KVCache* cache, float** out_values, size_t* out_seq_len) {
    if (!cache || !out_values || !out_seq_len) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Return pointers and current sequence length
    *out_values = cache->value_cache;
    *out_seq_len = cache->current_seq_len;
    
    return ONEBIT_SUCCESS;
} 