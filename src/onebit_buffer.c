#include "onebit/onebit_buffer.h"
#include "onebit/onebit_error.h"
#include <string.h>

int buffer_init(BufferContext* ctx, const BufferConfig* config) {
    if (!ctx || !config || config->initial_size == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->data = malloc(config->initial_size);
    if (!ctx->data) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    ctx->size = 0;
    ctx->capacity = config->initial_size;
    ctx->growth_factor = config->growth_factor > 1.0f ?
                        config->growth_factor : 2.0f;
    
    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        free(ctx->data);
        return ONEBIT_ERROR_THREAD;
    }
    
    return ONEBIT_SUCCESS;
}

void buffer_cleanup(BufferContext* ctx) {
    if (!ctx) return;
    
    free(ctx->data);
    pthread_mutex_destroy(&ctx->mutex);
}

static int buffer_grow(BufferContext* ctx, size_t min_size) {
    size_t new_capacity = ctx->capacity;
    
    while (new_capacity < min_size) {
        new_capacity = (size_t)(new_capacity * ctx->growth_factor);
    }
    
    void* new_data = realloc(ctx->data, new_capacity);
    if (!new_data) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    ctx->data = new_data;
    ctx->capacity = new_capacity;
    return ONEBIT_SUCCESS;
}

int buffer_append(BufferContext* ctx, const void* data,
                 size_t size) {
    if (!ctx || !data || size == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    // Check if buffer needs to grow
    if (ctx->size + size > ctx->capacity) {
        int result = buffer_grow(ctx, ctx->size + size);
        if (result != ONEBIT_SUCCESS) {
            pthread_mutex_unlock(&ctx->mutex);
            return result;
        }
    }
    
    // Append data
    memcpy((uint8_t*)ctx->data + ctx->size, data, size);
    ctx->size += size;
    
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
}

int buffer_insert(BufferContext* ctx, size_t offset,
                 const void* data, size_t size) {
    if (!ctx || !data || size == 0 || offset > ctx->size) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    // Check if buffer needs to grow
    if (ctx->size + size > ctx->capacity) {
        int result = buffer_grow(ctx, ctx->size + size);
        if (result != ONEBIT_SUCCESS) {
            pthread_mutex_unlock(&ctx->mutex);
            return result;
        }
    }
    
    // Insert data
    memcpy((uint8_t*)ctx->data + offset, data, size);
    ctx->size += size;
    
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
} 