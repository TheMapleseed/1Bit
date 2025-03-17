#ifndef ONEBIT_MEMORY_H
#define ONEBIT_MEMORY_H

#include <stddef.h>
#include <stdbool.h>

// Memory block flags
#define MEMORY_FLAG_PINNED     (1 << 0)
#define MEMORY_FLAG_DEVICE     (1 << 1)
#define MEMORY_FLAG_UNIFIED    (1 << 2)
#define MEMORY_FLAG_TEMPORARY  (1 << 3)
#define MEMORY_FLAG_PERSISTENT (1 << 4)

// Memory pool configuration
typedef struct {
    size_t initial_size;
    size_t max_size;
    size_t block_size;
    size_t alignment;
    bool enable_defrag;
    float growth_factor;
    int device_id;
} MemoryPoolConfig;

// Memory pool context
typedef struct {
    void* base_ptr;
    size_t total_size;
    size_t used_size;
    size_t peak_usage;
    MemoryPoolConfig config;
    void* free_list;
    pthread_mutex_t pool_mutex;
    void* device_context;
} MemoryPool;

// Function declarations
int memory_pool_init(MemoryPool* pool, const MemoryPoolConfig* config);
void memory_pool_cleanup(MemoryPool* pool);

void* memory_allocate(MemoryPool* pool, size_t size, uint32_t flags);
void memory_free(MemoryPool* pool, void* ptr);
void* memory_resize(MemoryPool* pool, void* ptr, size_t new_size);

// Memory transfer operations
int memory_copy_to_device(MemoryPool* pool, void* dst, const void* src, size_t size);
int memory_copy_to_host(MemoryPool* pool, void* dst, const void* src, size_t size);
int memory_copy_device_to_device(MemoryPool* pool, void* dst, const void* src, size_t size);

// Memory pool management
int memory_pool_defrag(MemoryPool* pool);
void memory_pool_reset(MemoryPool* pool);
size_t memory_pool_available(MemoryPool* pool);
void memory_pool_get_stats(MemoryPool* pool, size_t* total, size_t* used, size_t* peak);

#endif 