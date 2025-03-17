#include "onebit/onebit_recovery.h"
#include "onebit/onebit_error.h"
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif

int recovery_init(RecoveryContext* ctx, const char* checkpoint_dir) {
    if (!ctx || !checkpoint_dir) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    strncpy(ctx->checkpoint_dir, checkpoint_dir, sizeof(ctx->checkpoint_dir) - 1);
    ctx->max_checkpoints = 5;  // Default value
    ctx->checkpoint_interval = 300;  // 5 minutes
    ctx->latest_checkpoint = NULL;
    ctx->auto_recovery = false;
    ctx->crash_handler = NULL;
    ctx->crash_handler_ctx = NULL;
    
    // Create checkpoint directory if it doesn't exist
    struct stat st = {0};
    if (stat(checkpoint_dir, &st) == -1) {
        if (mkdir(checkpoint_dir, 0700) != 0) {
            return ONEBIT_ERROR_IO;
        }
    }
    
    return ONEBIT_SUCCESS;
}

void recovery_cleanup(RecoveryContext* ctx) {
    if (ctx->latest_checkpoint) {
        free(ctx->latest_checkpoint->model_state);
        free(ctx->latest_checkpoint);
    }
}

int create_checkpoint(RecoveryContext* ctx, const OneBitContext* model_ctx) {
    if (!ctx || !model_ctx) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    Checkpoint* checkpoint = malloc(sizeof(Checkpoint));
    if (!checkpoint) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Save model state
    checkpoint->state_size = /* calculate state size */;
    checkpoint->model_state = malloc(checkpoint->state_size);
    if (!checkpoint->model_state) {
        free(checkpoint);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Copy model state
    memcpy(checkpoint->model_state, /* model state source */, checkpoint->state_size);
    
    // Set metadata
    checkpoint->timestamp = time(NULL);
    checkpoint->checkpoint_id = /* generate unique ID */;
    snprintf(checkpoint->metadata, sizeof(checkpoint->metadata),
             "Checkpoint created at %s", ctime(&checkpoint->timestamp));
    
    // Update latest checkpoint
    if (ctx->latest_checkpoint) {
        free(ctx->latest_checkpoint->model_state);
        free(ctx->latest_checkpoint);
    }
    ctx->latest_checkpoint = checkpoint;
    
    return ONEBIT_SUCCESS;
}

int restore_from_checkpoint(RecoveryContext* ctx, OneBitContext* model_ctx) {
    if (!ctx || !model_ctx || !ctx->latest_checkpoint) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Restore model state
    memcpy(/* model state destination */,
           ctx->latest_checkpoint->model_state,
           ctx->latest_checkpoint->state_size);
    
    return ONEBIT_SUCCESS;
}

int enable_auto_recovery(RecoveryContext* ctx, bool enabled) {
    if (!ctx) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->auto_recovery = enabled;
    return ONEBIT_SUCCESS;
} 