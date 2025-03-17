#include "onebit/onebit_array.h"
#include "onebit/onebit_error.h"
#include <string.h>

int array_init(ArrayContext* ctx, const ArrayConfig* config) {
    if (!ctx || !config || config->element_size == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->data = malloc(config->initial_capacity * config->element_size);
    if (!ctx->data) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    ctx->size = 0;
    ctx->capacity = config->initial_capacity;
    ctx->element_size = config->element_size;
    ctx->growth_factor = config->growth_factor > 1.0f ?
                        config->growth_factor : 2.0f;
    
    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        free(ctx->data);
        return ONEBIT_ERROR_THREAD;
    }
    
    return ONEBIT_SUCCESS;
}

void array_cleanup(ArrayContext* ctx) {
    if (!ctx) return;
    
    free(ctx->data);
    pthread_mutex_destroy(&ctx->mutex);
}

static int array_grow(ArrayContext* ctx, size_t min_capacity) {
    size_t new_capacity = ctx->capacity;
    
    while (new_capacity < min_capacity) {
        new_capacity = (size_t)(new_capacity * ctx->growth_factor);
    }
    
    void* new_data = realloc(ctx->data,
                            new_capacity * ctx->element_size);
    if (!new_data) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    ctx->data = new_data;
    ctx->capacity = new_capacity;
    return ONEBIT_SUCCESS;
}

int array_push(ArrayContext* ctx, const void* element) {
    if (!ctx || !element) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    if (ctx->size >= ctx->capacity) {
        int result = array_grow(ctx, ctx->size + 1);
        if (result != ONEBIT_SUCCESS) {
            pthread_mutex_unlock(&ctx->mutex);
            return result;
        }
    }
    
    memcpy((uint8_t*)ctx->data + ctx->size * ctx->element_size,
           element, ctx->element_size);
    
    ctx->size++;
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
} 