#include <onebit/onebit_memory.h>
#include <onebit/onebit_error.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

// Memory pool structure
struct OneBitMemoryPoolStruct {
    MemoryAllocation* allocations;
    size_t num_allocations;
    size_t capacity;
    size_t total_size;
    size_t used_size;
    size_t peak_size;
    MemoryPoolConfig config;
    pthread_mutex_t mutex;
};

// Initialize memory pool
int memory_pool_init(OneBitMemoryPool** pool, const MemoryPoolConfig* config) {
    if (!pool || !config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    OneBitMemoryPool* new_pool = malloc(sizeof(OneBitMemoryPool));
    if (!new_pool) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Copy configuration
    memcpy(&new_pool->config, config, sizeof(MemoryPoolConfig));
    
    // Initialize allocations array
    new_pool->capacity = 1024;  // Initial capacity for allocations
    new_pool->allocations = malloc(new_pool->capacity * sizeof(MemoryAllocation));
    if (!new_pool->allocations) {
        free(new_pool);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Initialize other fields
    new_pool->num_allocations = 0;
    new_pool->total_size = 0;
    new_pool->used_size = 0;
    new_pool->peak_size = 0;
    
    // Initialize mutex
    if (pthread_mutex_init(&new_pool->mutex, NULL) != 0) {
        free(new_pool->allocations);
        free(new_pool);
        return ONEBIT_ERROR_THREAD;
    }
    
    *pool = new_pool;
    return ONEBIT_SUCCESS;
}

// Cleanup memory pool
void memory_pool_cleanup(OneBitMemoryPool* pool) {
    if (!pool) return;
    
    pthread_mutex_lock(&pool->mutex);
    
    // Free all allocations
    for (size_t i = 0; i < pool->num_allocations; i++) {
        if (!pool->allocations[i].is_free && pool->allocations[i].ptr) {
            // Free based on flags
            if (pool->allocations[i].flags & MEMORY_FLAG_DEVICE) {
                // If it's device memory, free using direct device free
                memory_direct_free(pool->allocations[i].ptr, 
                                 pool->allocations[i].flags,
                                 pool->allocations[i].device_id);
            } else {
                free(pool->allocations[i].ptr);
            }
        }
    }
    
    // Free the allocations array
    free(pool->allocations);
    
    pthread_mutex_unlock(&pool->mutex);
    pthread_mutex_destroy(&pool->mutex);
    
    // Free the pool itself
    free(pool);
}

// Find an existing allocation by pointer
static int find_allocation(OneBitMemoryPool* pool, void* ptr) {
    for (size_t i = 0; i < pool->num_allocations; i++) {
        if (pool->allocations[i].ptr == ptr) {
            return (int)i;
        }
    }
    return -1;
}

// Find or create a free allocation slot
static int find_free_allocation(OneBitMemoryPool* pool) {
    // First look for existing free slots
    for (size_t i = 0; i < pool->num_allocations; i++) {
        if (pool->allocations[i].is_free) {
            return (int)i;
        }
    }
    
    // If no free slots, expand the allocations array if needed
    if (pool->num_allocations >= pool->capacity) {
        size_t new_capacity = pool->capacity * 2;
        MemoryAllocation* new_allocations = realloc(pool->allocations, 
                                                   new_capacity * sizeof(MemoryAllocation));
        if (!new_allocations) {
            return -1;
        }
        
        pool->allocations = new_allocations;
        pool->capacity = new_capacity;
    }
    
    // Use the next available slot
    return (int)pool->num_allocations++;
}

// Memory allocation
void* memory_allocate(OneBitMemoryPool* pool, size_t size, int flags) {
    if (!pool || size == 0) {
        return NULL;
    }
    
    pthread_mutex_lock(&pool->mutex);
    
    // Check if we've reached the maximum pool size
    if (pool->config.max_size > 0 && 
        pool->used_size + size > pool->config.max_size && 
        !pool->config.allow_growth) {
        pthread_mutex_unlock(&pool->mutex);
        error_set(ONEBIT_ERROR_MEMORY, "Memory pool size limit exceeded");
        return NULL;
    }
    
    // Find a free allocation slot
    int index = find_free_allocation(pool);
    if (index < 0) {
        pthread_mutex_unlock(&pool->mutex);
        error_set(ONEBIT_ERROR_MEMORY, "Failed to find free allocation slot");
        return NULL;
    }
    
    // Allocate memory
    void* ptr = NULL;
    int device_id = (flags & MEMORY_FLAG_DEVICE) ? pool->config.device_id : -1;
    
    if (flags & MEMORY_FLAG_DEVICE) {
        // Device memory allocation
        ptr = memory_direct_allocate(size, flags, device_id);
    } else {
        // Host memory allocation
        ptr = malloc(size);
    }
    
    if (!ptr) {
        // Mark the allocation slot as free
        pool->allocations[index].is_free = true;
        pthread_mutex_unlock(&pool->mutex);
        error_set(ONEBIT_ERROR_MEMORY, "Failed to allocate memory");
        return NULL;
    }
    
    // Store allocation information
    pool->allocations[index].ptr = ptr;
    pool->allocations[index].size = size;
    pool->allocations[index].flags = flags;
    pool->allocations[index].device_id = device_id;
    pool->allocations[index].is_free = false;
    
    // Update pool statistics
    pool->used_size += size;
    if (pool->used_size > pool->peak_size) {
        pool->peak_size = pool->used_size;
    }
    
    pthread_mutex_unlock(&pool->mutex);
    return ptr;
}

// Memory free
void memory_free(OneBitMemoryPool* pool, void* ptr) {
    if (!pool || !ptr) {
        return;
    }
    
    pthread_mutex_lock(&pool->mutex);
    
    // Find the allocation
    int index = find_allocation(pool, ptr);
    if (index < 0) {
        pthread_mutex_unlock(&pool->mutex);
        error_set(ONEBIT_ERROR_INVALID_PARAM, "Pointer not found in memory pool");
        return;
    }
    
    // Skip if already freed
    if (pool->allocations[index].is_free) {
        pthread_mutex_unlock(&pool->mutex);
        return;
    }
    
    // Free the memory
    if (pool->allocations[index].flags & MEMORY_FLAG_DEVICE) {
        // Device memory
        memory_direct_free(ptr, pool->allocations[index].flags, 
                         pool->allocations[index].device_id);
    } else {
        // Host memory
        free(ptr);
    }
    
    // Update allocation information
    pool->used_size -= pool->allocations[index].size;
    pool->allocations[index].is_free = true;
    
    pthread_mutex_unlock(&pool->mutex);
}

// Memory reallocation
void* memory_reallocate(OneBitMemoryPool* pool, void* ptr, size_t new_size) {
    if (!pool) {
        return NULL;
    }
    
    // Handle NULL pointer as a new allocation
    if (!ptr) {
        return memory_allocate(pool, new_size, MEMORY_FLAG_DEFAULT);
    }
    
    // Handle zero size as a free operation
    if (new_size == 0) {
        memory_free(pool, ptr);
        return NULL;
    }
    
    pthread_mutex_lock(&pool->mutex);
    
    // Find the allocation
    int index = find_allocation(pool, ptr);
    if (index < 0) {
        pthread_mutex_unlock(&pool->mutex);
        error_set(ONEBIT_ERROR_INVALID_PARAM, "Pointer not found in memory pool");
        return NULL;
    }
    
    // Check if already freed
    if (pool->allocations[index].is_free) {
        pthread_mutex_unlock(&pool->mutex);
        error_set(ONEBIT_ERROR_INVALID_PARAM, "Attempt to reallocate freed memory");
        return NULL;
    }
    
    // Get current allocation info
    size_t old_size = pool->allocations[index].size;
    int flags = pool->allocations[index].flags;
    int device_id = pool->allocations[index].device_id;
    
    // Check if we've reached the maximum pool size
    if (pool->config.max_size > 0 && 
        pool->used_size - old_size + new_size > pool->config.max_size && 
        !pool->config.allow_growth) {
        pthread_mutex_unlock(&pool->mutex);
        error_set(ONEBIT_ERROR_MEMORY, "Memory pool size limit exceeded");
        return NULL;
    }
    
    // Reallocate memory
    void* new_ptr = NULL;
    
    if (flags & MEMORY_FLAG_DEVICE) {
        // Device memory reallocation (allocate new, copy, free old)
        new_ptr = memory_direct_allocate(new_size, flags, device_id);
        if (new_ptr) {
            // Copy data
            memory_copy(new_ptr, ptr, old_size < new_size ? old_size : new_size, 
                      flags, flags);
            memory_direct_free(ptr, flags, device_id);
        }
    } else {
        // Host memory reallocation
        new_ptr = realloc(ptr, new_size);
    }
    
    if (!new_ptr) {
        pthread_mutex_unlock(&pool->mutex);
        error_set(ONEBIT_ERROR_MEMORY, "Failed to reallocate memory");
        return NULL;
    }
    
    // Update allocation information
    pool->allocations[index].ptr = new_ptr;
    pool->allocations[index].size = new_size;
    
    // Update pool statistics
    pool->used_size = pool->used_size - old_size + new_size;
    if (pool->used_size > pool->peak_size) {
        pool->peak_size = pool->used_size;
    }
    
    pthread_mutex_unlock(&pool->mutex);
    return new_ptr;
}

// Temporary allocation
void* memory_allocate_temp(OneBitMemoryPool* pool, size_t size, int flags) {
    // Temporary allocations are just regular allocations with the TEMPORARY flag set
    return memory_allocate(pool, size, flags | MEMORY_FLAG_TEMPORARY);
}

// Copy memory
int memory_copy(void* dst, const void* src, size_t size, int dst_flags, int src_flags) {
    if (!dst || !src || size == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Host to Host
    if (!(dst_flags & MEMORY_FLAG_DEVICE) && !(src_flags & MEMORY_FLAG_DEVICE)) {
        memcpy(dst, src, size);
        return ONEBIT_SUCCESS;
    }
    
    // Host to Device, Device to Host, Device to Device
    // In a real implementation, this would use cudaMemcpy or equivalent
    // For this placeholder implementation, just do a memcpy
    memcpy(dst, src, size);
    
    return ONEBIT_SUCCESS;
}

// Get memory pool statistics
int memory_get_stats(OneBitMemoryPool* pool, size_t* total, size_t* used, size_t* peak) {
    if (!pool) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&pool->mutex);
    
    if (total) *total = pool->config.max_size;
    if (used) *used = pool->used_size;
    if (peak) *peak = pool->peak_size;
    
    pthread_mutex_unlock(&pool->mutex);
    
    return ONEBIT_SUCCESS;
}

// Print memory pool statistics
void memory_print_stats(OneBitMemoryPool* pool) {
    if (!pool) return;
    
    pthread_mutex_lock(&pool->mutex);
    
    printf("Memory Pool Statistics:\n");
    printf("  Total Size: %zu bytes\n", pool->config.max_size);
    printf("  Used Size: %zu bytes (%.2f%%)\n", 
           pool->used_size, 
           (double)pool->used_size / (double)pool->config.max_size * 100.0);
    printf("  Peak Size: %zu bytes (%.2f%%)\n", 
           pool->peak_size, 
           (double)pool->peak_size / (double)pool->config.max_size * 100.0);
    printf("  Allocations: %zu / %zu\n", 
           pool->num_allocations, pool->capacity);
    
    pthread_mutex_unlock(&pool->mutex);
}

// Clear all temporary allocations
void memory_clear_temp(OneBitMemoryPool* pool) {
    if (!pool) return;
    
    pthread_mutex_lock(&pool->mutex);
    
    // Free all temporary allocations
    for (size_t i = 0; i < pool->num_allocations; i++) {
        if (!pool->allocations[i].is_free && 
            (pool->allocations[i].flags & MEMORY_FLAG_TEMPORARY)) {
            // Free the memory
            if (pool->allocations[i].flags & MEMORY_FLAG_DEVICE) {
                memory_direct_free(pool->allocations[i].ptr, 
                                  pool->allocations[i].flags,
                                  pool->allocations[i].device_id);
            } else {
                free(pool->allocations[i].ptr);
            }
            
            // Update allocation information
            pool->used_size -= pool->allocations[i].size;
            pool->allocations[i].is_free = true;
        }
    }
    
    pthread_mutex_unlock(&pool->mutex);
}

// Direct memory allocation (without pool)
void* memory_direct_allocate(size_t size, int flags, int device_id) {
    if (size == 0) {
        return NULL;
    }
    
    if (flags & MEMORY_FLAG_DEVICE) {
        // Device memory allocation
        // In a real implementation, this would use cudaMalloc or equivalent
        // For this placeholder implementation, just use regular malloc
        return malloc(size);
    } else if (flags & MEMORY_FLAG_PINNED) {
        // Pinned memory allocation
        // In a real implementation, this would use cudaHostAlloc or equivalent
        // For this placeholder implementation, just use regular malloc
        return malloc(size);
    } else {
        // Regular host memory
        return malloc(size);
    }
}

// Direct memory free (without pool)
void memory_direct_free(void* ptr, int flags, int device_id) {
    if (!ptr) return;
    
    if (flags & MEMORY_FLAG_DEVICE) {
        // Device memory free
        // In a real implementation, this would use cudaFree or equivalent
        // For this placeholder implementation, just use regular free
        free(ptr);
    } else if (flags & MEMORY_FLAG_PINNED) {
        // Pinned memory free
        // In a real implementation, this would use cudaFreeHost or equivalent
        // For this placeholder implementation, just use regular free
        free(ptr);
    } else {
        // Regular host memory
        free(ptr);
    }
} 