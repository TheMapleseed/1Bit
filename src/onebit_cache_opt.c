/**
 * @file onebit_cache_opt.c
 * @brief Implementation of high-performance caching system
 */

#include "onebit/onebit_cache_opt.h"
#include "onebit/onebit_error.h"
#include <stdlib.h>
#include <string.h>

// Get set index from address
static inline size_t get_set_index(const CacheContext* ctx, const void* addr) {
    return ((uintptr_t)addr / ctx->config.block_size) % ctx->num_sets;
}

// Get tag from address
static inline uint64_t get_tag(const CacheContext* ctx, const void* addr) {
    return ((uintptr_t)addr / ctx->config.block_size) / ctx->num_sets;
}

// Find block in set
static CacheBlock* find_block(CacheContext* ctx, size_t set_idx, uint64_t tag) {
    CacheBlock* set = ctx->blocks[set_idx];
    
    for (size_t i = 0; i < ctx->num_ways; i++) {
        if (set[i].state != CACHE_INVALID && set[i].tag == tag) {
            set[i].timestamp = ++ctx->clock;
            return &set[i];
        }
    }
    return NULL;
}

// Find victim block for replacement
static CacheBlock* find_victim(CacheContext* ctx, size_t set_idx) {
    CacheBlock* set = ctx->blocks[set_idx];
    CacheBlock* victim = &set[0];
    uint64_t oldest = set[0].timestamp;
    
    // First check for invalid blocks
    for (size_t i = 0; i < ctx->num_ways; i++) {
        if (set[i].state == CACHE_INVALID) {
            return &set[i];
        }
    }
    
    // Otherwise find least recently used
    for (size_t i = 1; i < ctx->num_ways; i++) {
        if (set[i].timestamp < oldest) {
            oldest = set[i].timestamp;
            victim = &set[i];
        }
    }
    
    return victim;
}

int cache_init(CacheContext* ctx, const CacheConfig* config) {
    if (!ctx || !config) return ONEBIT_ERROR_INVALID_PARAM;
    
    // Validate configuration
    if (config->total_size == 0 || config->block_size == 0 || 
        config->ways == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Calculate cache organization
    ctx->num_sets = config->total_size / (config->block_size * config->ways);
    ctx->num_ways = config->ways;
    memcpy(&ctx->config, config, sizeof(CacheConfig));
    
    // Allocate cache blocks
    ctx->blocks = malloc(ctx->num_sets * sizeof(CacheBlock*));
    if (!ctx->blocks) return ONEBIT_ERROR_MEMORY;
    
    for (size_t i = 0; i < ctx->num_sets; i++) {
        ctx->blocks[i] = calloc(ctx->num_ways, sizeof(CacheBlock));
        if (!ctx->blocks[i]) {
            for (size_t j = 0; j < i; j++) {
                for (size_t k = 0; k < ctx->num_ways; k++) {
                    free(ctx->blocks[j][k].data);
                }
                free(ctx->blocks[j]);
            }
            free(ctx->blocks);
            return ONEBIT_ERROR_MEMORY;
        }
        
        for (size_t j = 0; j < ctx->num_ways; j++) {
            ctx->blocks[i][j].data = malloc(config->block_size);
            if (!ctx->blocks[i][j].data) {
                for (size_t k = 0; k < j; k++) {
                    free(ctx->blocks[i][k].data);
                }
                for (size_t k = 0; k < i; k++) {
                    for (size_t l = 0; l < ctx->num_ways; l++) {
                        free(ctx->blocks[k][l].data);
                    }
                    free(ctx->blocks[k]);
                }
                free(ctx->blocks[i]);
                free(ctx->blocks);
                return ONEBIT_ERROR_MEMORY;
            }
        }
    }
    return ONEBIT_SUCCESS;
} 