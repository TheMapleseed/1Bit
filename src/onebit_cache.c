#include "onebit/onebit_cache.h"
#include "onebit/onebit_error.h"
#include <string.h>

// Cache entry
typedef struct CacheEntry {
    char* key;
    void* data;
    size_t size;
    uint64_t last_access;
    struct CacheEntry* next;
} CacheEntry;

int cache_init(CacheContext* ctx, const CacheConfig* config) {
    if (!ctx || !config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->entries = calloc(config->num_buckets, sizeof(CacheEntry*));
    if (!ctx->entries) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    ctx->num_buckets = config->num_buckets;
    ctx->max_entries = config->max_entries;
    ctx->max_memory = config->max_memory;
    ctx->num_entries = 0;
    ctx->total_memory = 0;
    ctx->access_count = 0;
    
    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        free(ctx->entries);
        return ONEBIT_ERROR_THREAD;
    }
    
    return ONEBIT_SUCCESS;
}

void cache_cleanup(CacheContext* ctx) {
    if (!ctx) return;
    
    pthread_mutex_lock(&ctx->mutex);
    
    // Free all entries
    for (size_t i = 0; i < ctx->num_buckets; i++) {
        CacheEntry* entry = ctx->entries[i];
        while (entry) {
            CacheEntry* next = entry->next;
            free(entry->key);
            free(entry->data);
            free(entry);
            entry = next;
        }
    }
    
    free(ctx->entries);
    
    pthread_mutex_unlock(&ctx->mutex);
    pthread_mutex_destroy(&ctx->mutex);
}

static uint32_t hash_key(const char* key) {
    uint32_t hash = 5381;
    int c;
    
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;
    }
    
    return hash;
}

static void evict_entries(CacheContext* ctx) {
    // Simple LRU eviction
    while ((ctx->num_entries > ctx->max_entries) ||
           (ctx->max_memory > 0 && ctx->total_memory > ctx->max_memory)) {
        
        uint64_t oldest_access = UINT64_MAX;
        CacheEntry* oldest_entry = NULL;
        
        for (size_t i = 0; i < ctx->num_buckets; i++) {
            CacheEntry* entry = ctx->entries[i];
            while (entry) {
                if (entry->last_access < oldest_access) {
                    oldest_access = entry->last_access;
                    oldest_entry = entry;
                }
                entry = entry->next;
            }
        }
        
        if (oldest_entry) {
            CacheEntry* next = oldest_entry->next;
            free(oldest_entry->key);
            free(oldest_entry->data);
            free(oldest_entry);
            oldest_entry = next;
        }
    }
} 