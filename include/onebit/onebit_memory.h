/**
 * @file onebit_memory.h
 * @brief Memory management for OneBit
 *
 * Provides efficient memory allocation, pooling, and management
 * for neural network operations.
 *
 * @author OneBit Team
 * @version 1.0.0
 */

#ifndef ONEBIT_onebit_memory_H
#define ONEBIT_onebit_memory_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Memory flags
#define MEMORY_FLAG_DEFAULT    0x00
#define ONEBIT_onebit_memory_HOST       0x01
#define MEMORY_FLAG_DEVICE     0x02
#define MEMORY_FLAG_PINNED     0x04
#define MEMORY_FLAG_MANAGED    0x08
#define MEMORY_FLAG_TEMPORARY  0x10

// Forward declaration
struct OneBitMemoryPoolStruct;
typedef struct OneBitMemoryPoolStruct OneBitMemoryPool;

// Memory allocation info
typedef struct {
    void* ptr;
    size_t size;
    int flags;
    int device_id;
    bool is_free;
} MemoryAllocation;

// Memory pool configuration
typedef struct {
    size_t initial_size;
    size_t max_size;
    bool allow_growth;
    bool use_device_memory;
    int device_id;
    int flags;
} MemoryPoolConfig;

// Initialize memory pool
int memory_pool_init(OneBitMemoryPool** pool, const MemoryPoolConfig* config);

// Cleanup memory pool
void memory_pool_cleanup(OneBitMemoryPool* pool);

// Memory allocation
void* memory_allocate(OneBitMemoryPool* pool, size_t size, int flags);

// Memory free
void memory_free(OneBitMemoryPool* pool, void* ptr);

// Memory reallocation
void* memory_reallocate(OneBitMemoryPool* pool, void* ptr, size_t new_size);

// Temporary allocation (automatically freed at synchronization point)
void* memory_allocate_temp(OneBitMemoryPool* pool, size_t size, int flags);

// Copy memory
int memory_copy(void* dst, const void* src, size_t size, 
               int dst_flags, int src_flags);

// Get memory pool statistics
int memory_get_stats(OneBitMemoryPool* pool, size_t* total, size_t* used, size_t* peak);

// Print memory pool statistics
void memory_print_stats(OneBitMemoryPool* pool);

// Clear all temporary allocations
void memory_clear_temp(OneBitMemoryPool* pool);

// Direct memory allocation (without pool)
void* memory_direct_allocate(size_t size, int flags, int device_id);

// Direct memory free (without pool)
void memory_direct_free(void* ptr, int flags, int device_id);

#ifdef __cplusplus
}
#endif

#endif /* ONEBIT_MEMORY_H */ 