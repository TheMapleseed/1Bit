/**
 * @file onebit_memory_opt.h
 * @brief High-performance memory management system
 *
 * Provides optimized memory allocation, pooling, and management
 * with support for NUMA awareness, huge pages, and aligned access.
 */

#ifndef ONEBIT_MEMORY_OPT_H
#define ONEBIT_MEMORY_OPT_H

#include <stddef.h>
#include <stdbool.h>

// Memory pool configuration
typedef struct {
    size_t block_size;
    size_t max_blocks;
    bool use_huge_pages;
    bool numa_aware;
    int numa_node;
    size_t alignment;
    bool enable_tracking;
} MemoryPoolConfig;

// Memory pool
typedef struct {
    void* blocks;
    void* free_list;
    size_t used_blocks;
    size_t total_blocks;
    MemoryPoolConfig config;
    void* mutex;
    void* stats;
} MemoryPool;

// Initialize and cleanup
int memory_pool_init(MemoryPool* pool, const MemoryPoolConfig* config);
void memory_pool_cleanup(MemoryPool* pool);

// Memory operations
void* memory_alloc(MemoryPool* pool, size_t size);
void memory_free(MemoryPool* pool, void* ptr);
void* memory_realloc(MemoryPool* pool, void* ptr, size_t new_size);

// Utility functions
size_t memory_get_allocated_size(const MemoryPool* pool);
void memory_print_stats(const MemoryPool* pool);

#endif /* ONEBIT_MEMORY_OPT_H */ 