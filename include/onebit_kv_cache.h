/**
 * @file onebit_kv_cache.h
 * @brief Key-Value cache implementation for efficient incremental decoding
 */

#ifndef ONEBIT_KV_CACHE_H
#define ONEBIT_KV_CACHE_H

#include "onebit_context.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief KV cache structure for efficient incremental decoding
 */
typedef struct {
    /** Key cache buffer */
    float* key_cache;
    
    /** Value cache buffer */
    float* value_cache;
    
    /** Current sequence length (number of tokens cached) */
    size_t current_seq_len;
    
    /** Maximum sequence length supported by this cache */
    size_t max_seq_len;
    
    /** Number of attention heads */
    size_t num_heads;
    
    /** Size of each attention head */
    size_t head_dim;
    
    /** Batch size */
    size_t batch_size;
} KVCache;

/**
 * @brief Initialize a KV cache
 * 
 * @param ctx Context for memory management
 * @param cache Pointer to KV cache structure to initialize
 * @param batch_size Batch size for inference
 * @param num_heads Number of attention heads
 * @param head_dim Size of each attention head
 * @param max_seq_len Maximum sequence length
 * @return ONEBIT_SUCCESS on success, error code otherwise
 */
int kv_cache_init(OneBitContext* ctx, KVCache* cache,
                size_t batch_size, size_t num_heads,
                size_t head_dim, size_t max_seq_len);

/**
 * @brief Free a KV cache
 * 
 * @param ctx Context for memory management
 * @param cache Pointer to KV cache structure to free
 */
void kv_cache_free(OneBitContext* ctx, KVCache* cache);

/**
 * @brief Append new keys and values to the cache
 * 
 * @param ctx Context for memory management
 * @param cache Pointer to KV cache structure
 * @param new_keys New keys to append (shape: [batch_size, num_heads, seq_len, head_dim])
 * @param new_values New values to append (shape: [batch_size, num_heads, seq_len, head_dim])
 * @param new_seq_len Length of new sequence
 * @return ONEBIT_SUCCESS on success, error code otherwise
 */
int kv_cache_append(OneBitContext* ctx, KVCache* cache,
                  const float* new_keys, const float* new_values,
                  size_t new_seq_len);

/**
 * @brief Reset a KV cache to empty state
 * 
 * @param ctx Context for memory management
 * @param cache Pointer to KV cache structure
 * @return ONEBIT_SUCCESS on success, error code otherwise
 */
int kv_cache_reset(OneBitContext* ctx, KVCache* cache);

/**
 * @brief Get the cached keys
 * 
 * @param cache Pointer to KV cache structure
 * @param out_keys Output pointer for cached keys
 * @param out_seq_len Output pointer for current sequence length
 * @return ONEBIT_SUCCESS on success, error code otherwise
 */
int kv_cache_get_keys(const KVCache* cache, float** out_keys, size_t* out_seq_len);

/**
 * @brief Get the cached values
 * 
 * @param cache Pointer to KV cache structure
 * @param out_values Output pointer for cached values
 * @param out_seq_len Output pointer for current sequence length
 * @return ONEBIT_SUCCESS on success, error code otherwise
 */
int kv_cache_get_values(const KVCache* cache, float** out_values, size_t* out_seq_len);

#ifdef __cplusplus
}
#endif

#endif /* ONEBIT_KV_CACHE_H */ 