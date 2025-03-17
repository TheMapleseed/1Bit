#include "onebit/onebit_string.h"
#include "onebit/onebit_error.h"
#include <string.h>
#include <ctype.h>

int string_init(StringContext* ctx, const StringConfig* config) {
    if (!ctx) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    size_t initial_capacity = config ? config->initial_capacity : 16;
    if (initial_capacity == 0) {
        initial_capacity = 16;
    }
    
    ctx->data = malloc(initial_capacity);
    if (!ctx->data) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    ctx->data[0] = '\0';
    ctx->length = 0;
    ctx->capacity = initial_capacity;
    ctx->growth_factor = config && config->growth_factor > 1.0f ?
                        config->growth_factor : 2.0f;
    
    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        free(ctx->data);
        return ONEBIT_ERROR_THREAD;
    }
    
    return ONEBIT_SUCCESS;
}

void string_cleanup(StringContext* ctx) {
    if (!ctx) return;
    
    free(ctx->data);
    pthread_mutex_destroy(&ctx->mutex);
}

static int string_grow(StringContext* ctx, size_t min_capacity) {
    size_t new_capacity = ctx->capacity;
    
    while (new_capacity < min_capacity) {
        new_capacity = (size_t)(new_capacity * ctx->growth_factor);
    }
    
    char* new_data = realloc(ctx->data, new_capacity);
    if (!new_data) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    ctx->data = new_data;
    ctx->capacity = new_capacity;
    return ONEBIT_SUCCESS;
}

int string_append(StringContext* ctx, const char* str) {
    if (!ctx || !str) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    size_t len = strlen(str);
    if (len == 0) {
        return ONEBIT_SUCCESS;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    size_t new_length = ctx->length + len;
    if (new_length + 1 > ctx->capacity) {
        int result = string_grow(ctx, new_length + 1);
        if (result != ONEBIT_SUCCESS) {
            pthread_mutex_unlock(&ctx->mutex);
            return result;
        }
    }
    
    memcpy(ctx->data + ctx->length, str, len + 1);
    ctx->length = new_length;
    
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
}

int string_insert(StringContext* ctx, size_t pos,
                 const char* str) {
    if (!ctx || !str || pos > ctx->length) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    size_t len = strlen(str);
    if (len == 0) {
        return ONEBIT_SUCCESS;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    size_t new_length = ctx->length + len;
    if (new_length + 1 > ctx->capacity) {
        int result = string_grow(ctx, new_length + 1);
        if (result != ONEBIT_SUCCESS) {
            pthread_mutex_unlock(&ctx->mutex);
            return result;
        }
    }
    
    memmove(ctx->data + pos + len, ctx->data + pos,
            ctx->length - pos + 1);
    memcpy(ctx->data + pos, str, len);
    ctx->length = new_length;
    
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
}

int string_erase(StringContext* ctx, size_t pos, size_t len) {
    if (!ctx || pos >= ctx->length) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    if (len > ctx->length - pos) {
        len = ctx->length - pos;
    }
    
    memmove(ctx->data + pos, ctx->data + pos + len,
            ctx->length - pos - len + 1);
    ctx->length -= len;
    
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
}

int string_replace(StringContext* ctx, const char* old_str,
                  const char* new_str) {
    if (!ctx || !old_str || !new_str) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    size_t old_len = strlen(old_str);
    if (old_len == 0) {
        return ONEBIT_SUCCESS;
    }
    
    size_t new_len = strlen(new_str);
    if (new_len == 0) {
        return ONEBIT_SUCCESS;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    size_t pos = 0;
    while ((pos = strstr(ctx->data + pos, old_str) - ctx->data) != ctx->length) {
        memmove(ctx->data + pos, ctx->data + pos + old_len,
                ctx->length - pos - old_len + 1);
        memcpy(ctx->data + pos, new_str, new_len);
        ctx->length -= old_len - new_len;
        pos += new_len;
    }
    
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
} 