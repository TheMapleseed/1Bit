#include "onebit/onebit_checkpoint.h"
#include "onebit/onebit_error.h"
#include <string.h>
#include <time.h>

int checkpoint_init(CheckpointContext* ctx,
                   const CheckpointConfig* config) {
    if (!ctx || !config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->save_dir = strdup(config->save_dir);
    if (!ctx->save_dir) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    ctx->max_checkpoints = config->max_checkpoints;
    ctx->save_interval = config->save_interval;
    ctx->last_save = 0;
    ctx->step = 0;
    
    // Create checkpoint directory if it doesn't exist
    #ifdef _WIN32
    CreateDirectoryA(ctx->save_dir, NULL);
    #else
    mkdir(ctx->save_dir, 0755);
    #endif
    
    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        free(ctx->save_dir);
        return ONEBIT_ERROR_THREAD;
    }
    
    return ONEBIT_SUCCESS;
}

void checkpoint_cleanup(CheckpointContext* ctx) {
    if (!ctx) return;
    
    free(ctx->save_dir);
    pthread_mutex_destroy(&ctx->mutex);
}

static char* get_checkpoint_path(const CheckpointContext* ctx,
                               uint64_t step) {
    char* path = malloc(strlen(ctx->save_dir) + 64);
    if (!path) return NULL;
    
    snprintf(path, strlen(ctx->save_dir) + 64, "%s/checkpoint_%lu.bin",
             ctx->save_dir, step);
    
    return path;
}

int checkpoint_save(CheckpointContext* ctx, const void* state,
                   size_t state_size) {
    if (!ctx || !state || state_size == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    // Check if it's time to save
    if (ctx->step - ctx->last_save < ctx->save_interval) {
        pthread_mutex_unlock(&ctx->mutex);
        return ONEBIT_SUCCESS;
    }
    
    char* path = get_checkpoint_path(ctx, ctx->step);
    if (!path) {
        pthread_mutex_unlock(&ctx->mutex);
        return ONEBIT_ERROR_MEMORY;
    }
    
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        free(path);
        pthread_mutex_unlock(&ctx->mutex);
        return ONEBIT_ERROR_IO;
    }
    
    // Write header
    CheckpointHeader header;
    header.magic = CHECKPOINT_MAGIC;
    header.version = CHECKPOINT_VERSION;
    header.step = ctx->step;
    header.timestamp = time(NULL);
    header.state_size = state_size;
    
    if (fwrite(&header, sizeof(header), 1, fp) != 1) {
        fclose(fp);
        free(path);
        pthread_mutex_unlock(&ctx->mutex);
        return ONEBIT_ERROR_IO;
    }
    
    // Write state data
    if (fwrite(state, state_size, 1, fp) != 1) {
        fclose(fp);
        free(path);
        pthread_mutex_unlock(&ctx->mutex);
        return ONEBIT_ERROR_IO;
    }
    
    fclose(fp);
    
    // Update checkpoint tracking
    ctx->last_save = ctx->step;
    
    // Remove old checkpoints if needed
    if (ctx->max_checkpoints > 0) {
        char old_path[1024];
        uint64_t old_step = (ctx->step > ctx->max_checkpoints * ctx->save_interval) ?
                           ctx->step - ctx->max_checkpoints * ctx->save_interval : 0;
        while (old_step < ctx->step) {
            snprintf(old_path, sizeof(old_path), "%s/checkpoint_%lu.bin",
                     ctx->save_dir, old_step);
            remove(old_path);
            old_step += ctx->save_interval;
        }
    }
    
    pthread_mutex_unlock(&ctx->mutex);
    
    return ONEBIT_SUCCESS;
} 