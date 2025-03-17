#include "onebit/onebit_dataset.h"
#include "onebit/onebit_error.h"
#include <string.h>
#include <stdlib.h>

// Internal dataset entry
typedef struct {
    void* data;
    size_t size;
    DataType type;
    char* label;
} DatasetEntry;

int dataset_init(DatasetContext* ctx, const DatasetConfig* config) {
    if (!ctx || !config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->entries = malloc(config->initial_capacity * sizeof(DatasetEntry));
    if (!ctx->entries) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    ctx->capacity = config->initial_capacity;
    ctx->size = 0;
    memcpy(&ctx->config, config, sizeof(DatasetConfig));
    
    // Initialize mutex for thread safety
    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        free(ctx->entries);
        return ONEBIT_ERROR_THREAD;
    }
    
    return ONEBIT_SUCCESS;
}

void dataset_cleanup(DatasetContext* ctx) {
    if (!ctx) return;
    
    pthread_mutex_lock(&ctx->mutex);
    
    for (size_t i = 0; i < ctx->size; i++) {
        free(ctx->entries[i].data);
        free(ctx->entries[i].label);
    }
    
    free(ctx->entries);
    
    pthread_mutex_unlock(&ctx->mutex);
    pthread_mutex_destroy(&ctx->mutex);
}

int dataset_add(DatasetContext* ctx, const void* data,
                size_t size, DataType type, const char* label) {
    if (!ctx || !data || size == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    // Check if resize is needed
    if (ctx->size >= ctx->capacity) {
        size_t new_capacity = ctx->capacity * 2;
        DatasetEntry* new_entries = realloc(ctx->entries,
                                          new_capacity * sizeof(DatasetEntry));
        if (!new_entries) {
            pthread_mutex_unlock(&ctx->mutex);
            return ONEBIT_ERROR_MEMORY;
        }
        
        ctx->entries = new_entries;
        ctx->capacity = new_capacity;
    }
    
    // Allocate and copy data
    void* data_copy = malloc(size);
    if (!data_copy) {
        pthread_mutex_unlock(&ctx->mutex);
        return ONEBIT_ERROR_MEMORY;
    }
    
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
} 