/**
 * @file onebit_memory_opt.c
 * @brief Implementation of memory optimization system
 */

#include "onebit/onebit_memory_opt.h"
#include "onebit/onebit_error.h"
#include <pthread.h>
#include <string.h>
#include <sys/mman.h>
#include <numa.h>

// Memory block header
typedef struct MemoryBlock {
    size_t size;
    bool is_free;
    struct MemoryBlock* next;
    struct MemoryBlock* prev;
    size_t magic;  // For validation
    void* data;
} MemoryBlock;

#define MEMORY_BLOCK_MAGIC 0xDEADBEEF
#define MIN_BLOCK_SIZE 64
#define ALIGNMENT_MASK (~(size_t)(63))

static void* allocate_huge_pages(size_t size, int numa_node) {
    void* ptr = NULL;
    
    #ifdef MADV_HUGEPAGE
    // Try to allocate huge pages
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr != MAP_FAILED) {
        madvise(ptr, size, MADV_HUGEPAGE);
        
        if (numa_node >= 0) {
            numa_tonode_memory(ptr, size, numa_node);
        }
    }
    #endif
    
    return (ptr != MAP_FAILED) ? ptr : NULL;
}

int memory_pool_init(MemoryPool* pool, const MemoryPoolConfig* config) {
    if (!pool || !config) return ONEBIT_ERROR_INVALID;
    if (config->block_size < MIN_BLOCK_SIZE) return ONEBIT_ERROR_INVALID;
    
    memset(pool, 0, sizeof(MemoryPool));
    pool->config = *config;
    
    // Initialize mutex
    pool->mutex = malloc(sizeof(pthread_mutex_t));
    if (!pool->mutex) return ONEBIT_ERROR_MEMORY;
    pthread_mutex_init((pthread_mutex_t*)pool->mutex, NULL);
    
    // Calculate total size needed
    size_t total_size = config->block_size * config->max_blocks;
    
    // Allocate memory
    if (config->use_huge_pages) {
        pool->blocks = allocate_huge_pages(total_size, config->numa_node);
    } else {
        pool->blocks = aligned_alloc(config->alignment, total_size);
    }
    
    if (!pool->blocks) {
        pthread_mutex_destroy((pthread_mutex_t*)pool->mutex);
        free(pool->mutex);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Initialize free list
    MemoryBlock* current = (MemoryBlock*)pool->blocks;
    pool->free_list = current;
    
    for (size_t i = 0; i < config->max_blocks; i++) {
        current->size = config->block_size - sizeof(MemoryBlock);
        current->is_free = true;
        current->magic = MEMORY_BLOCK_MAGIC;
        current->data = (void*)(current + 1);
        
        if (i < config->max_blocks - 1) {
            current->next = (MemoryBlock*)((char*)current + config->block_size);
            current->next->prev = current;
        }
        
        current = current->next;
    }
    
    pool->total_blocks = config->max_blocks;
    pool->used_blocks = 0;
    
    // Initialize statistics if enabled
    if (config->enable_tracking) {
        pool->stats = calloc(1, sizeof(MemoryStats));
    }
    
    return ONEBIT_SUCCESS;
}

void* memory_alloc(MemoryPool* pool, size_t size) {
    if (!pool || size == 0) return NULL;
    
    pthread_mutex_lock((pthread_mutex_t*)pool->mutex);
    
    // Align size to boundary
    size = (size + pool->config.alignment - 1) & ALIGNMENT_MASK;
    
    // Find suitable block
    MemoryBlock* current = (MemoryBlock*)pool->free_list;
    while (current) {
        if (current->is_free && current->size >= size) {
            // Check if block can be split
            if (current->size >= size + sizeof(MemoryBlock) + MIN_BLOCK_SIZE) {
                MemoryBlock* new_block = (MemoryBlock*)((char*)current->data + size);
                new_block->size = current->size - size - sizeof(MemoryBlock);
                new_block->is_free = true;
                new_block->magic = MEMORY_BLOCK_MAGIC;
                new_block->data = (void*)(new_block + 1);
                
                // Update links
                new_block->next = current->next;
                new_block->prev = current;
                if (current->next) current->next->prev = new_block;
                current->next = new_block;
                
                current->size = size;
            }
            
            current->is_free = false;
            pool->used_blocks++;
            
            if (pool->stats) {
                ((MemoryStats*)pool->stats)->allocated_bytes += current->size;
                ((MemoryStats*)pool->stats)->allocation_count++;
            }
            
            pthread_mutex_unlock((pthread_mutex_t*)pool->mutex);
            return current->data;
        }
        current = current->next;
    }
    
    pthread_mutex_unlock((pthread_mutex_t*)pool->mutex);
    return NULL;
}

void memory_free(MemoryPool* pool, void* ptr) {
    if (!pool || !ptr) return;
    
    pthread_mutex_lock((pthread_mutex_t*)pool->mutex);
    
    // Get block header
    MemoryBlock* block = (MemoryBlock*)((char*)ptr - sizeof(MemoryBlock));
    
    // Validate block
    if (block->magic != MEMORY_BLOCK_MAGIC) {
        pthread_mutex_unlock((pthread_mutex_t*)pool->mutex);
        return;
    }
    
    block->is_free = true;
    pool->used_blocks--;
    
    // Coalesce with adjacent free blocks
    if (block->next && block->next->is_free) {
        block->size += sizeof(MemoryBlock) + block->next->size;
        block->next = block->next->next;
        if (block->next) block->next->prev = block;
    }
    
    if (block->prev && block->prev->is_free) {
        block->prev->size += sizeof(MemoryBlock) + block->size;
        block->prev->next = block->next;
        if (block->next) block->next->prev = block->prev;
    }
    
    if (pool->stats) {
        ((MemoryStats*)pool->stats)->freed_bytes += block->size;
        ((MemoryStats*)pool->stats)->free_count++;
    }
    
    pthread_mutex_unlock((pthread_mutex_t*)pool->mutex);
} 