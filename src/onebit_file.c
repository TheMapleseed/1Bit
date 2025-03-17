#include "onebit/onebit_file.h"
#include "onebit/onebit_error.h"
#include <string.h>

int file_open(FileContext* ctx, const char* filename,
              FileMode mode) {
    if (!ctx || !filename) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    const char* mode_str;
    switch (mode) {
        case FILE_MODE_READ:
            mode_str = "rb";
            break;
        case FILE_MODE_WRITE:
            mode_str = "wb";
            break;
        case FILE_MODE_APPEND:
            mode_str = "ab";
            break;
        default:
            return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->file = fopen(filename, mode_str);
    if (!ctx->file) {
        return ONEBIT_ERROR_IO;
    }
    
    ctx->mode = mode;
    strncpy(ctx->filename, filename, sizeof(ctx->filename) - 1);
    ctx->filename[sizeof(ctx->filename) - 1] = '\0';
    
    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        fclose(ctx->file);
        return ONEBIT_ERROR_THREAD;
    }
    
    return ONEBIT_SUCCESS;
}

void file_close(FileContext* ctx) {
    if (!ctx) return;
    
    pthread_mutex_lock(&ctx->mutex);
    
    if (ctx->file) {
        fclose(ctx->file);
        ctx->file = NULL;
    }
    
    pthread_mutex_unlock(&ctx->mutex);
    pthread_mutex_destroy(&ctx->mutex);
}

int file_read(FileContext* ctx, void* buffer, size_t size,
              size_t* bytes_read) {
    if (!ctx || !buffer || size == 0 || !bytes_read) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    if (ctx->mode != FILE_MODE_READ) {
        return ONEBIT_ERROR_INVALID_STATE;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    *bytes_read = fread(buffer, 1, size, ctx->file);
    
    if (*bytes_read < size && ferror(ctx->file)) {
        pthread_mutex_unlock(&ctx->mutex);
        return ONEBIT_ERROR_IO;
    }
    
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
}

int file_write(FileContext* ctx, const void* buffer,
               size_t size) {
    if (!ctx || !buffer || size == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    if (ctx->mode != FILE_MODE_WRITE &&
        ctx->mode != FILE_MODE_APPEND) {
        return ONEBIT_ERROR_INVALID_STATE;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    size_t written = fwrite(buffer, 1, size, ctx->file);
    
    if (written < size) {
        pthread_mutex_unlock(&ctx->mutex);
        return ONEBIT_ERROR_IO;
    }
    
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
}

int file_flush(FileContext* ctx) {
    if (!ctx) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    if (fflush(ctx->file) != 0) {
        pthread_mutex_unlock(&ctx->mutex);
        return ONEBIT_ERROR_IO;
    }
    
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
} 