/**
 * @file onebit_cache_opt.h
 * @brief High-performance caching system
 *
 * Implements efficient caching strategies for model weights,
 * activations, and computed results with LRU/MRU policies.
 */

#ifndef ONEBIT_CACHE_OPT_H
#define ONEBIT_CACHE_OPT_H

#include <stddef.h>
#include <stdbool.h>

// Cache entry types
typedef enum {
    CACHE_TYPE_WEIGHT,
    CACHE_TYPE_ACTIVATION,
    CACHE_TYPE_KV,
    CACHE_TYPE_RESULT
} CacheEntryType;

// Cache configuration
typedef struct {
    size_t max_size;
    size_t entry_size;
    bool enable_prefetch;
    bool enable_compression;
    int replacement_policy;
    float eviction_threshold;
} CacheConfig;

// Cache context
typedef struct {
    void* entries;
    void* index;
    size_t used_size;
    CacheConfig config;
    void* mutex;
    void* stats;
} CacheContext;

// Initialize and cleanup
int cache_init(CacheContext* ctx, const CacheConfig* config);
void cache_cleanup(CacheContext* ctx);

// Cache operations
int cache_store(CacheContext* ctx, const void* key, size_t key_size,
               const void* data, size_t data_size, CacheEntryType type);
int cache_lookup(CacheContext* ctx, const void* key, size_t key_size,
                void* data, size_t* data_size);
int cache_remove(CacheContext* ctx, const void* key, size_t key_size);
int cache_clear(CacheContext* ctx);

// Utility functions
size_t cache_get_size(const CacheContext* ctx);
void cache_print_stats(const CacheContext* ctx);

#endif /* ONEBIT_CACHE_OPT_H */ 