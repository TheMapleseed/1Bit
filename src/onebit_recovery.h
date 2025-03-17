#ifndef ONEBIT_RECOVERY_H
#define ONEBIT_RECOVERY_H

#include <stdbool.h>
#include <time.h>

// Checkpoint structure
typedef struct {
    char* model_state;
    size_t state_size;
    time_t timestamp;
    uint32_t checkpoint_id;
    char metadata[1024];
} Checkpoint;

// Recovery context
typedef struct {
    char checkpoint_dir[256];
    int max_checkpoints;
    int checkpoint_interval;  // seconds
    Checkpoint* latest_checkpoint;
    bool auto_recovery;
    void (*crash_handler)(void*);
    void* crash_handler_ctx;
} RecoveryContext;

// Function declarations
int recovery_init(RecoveryContext* ctx, const char* checkpoint_dir);
void recovery_cleanup(RecoveryContext* ctx);
int create_checkpoint(RecoveryContext* ctx, const OneBitContext* model_ctx);
int restore_from_checkpoint(RecoveryContext* ctx, OneBitContext* model_ctx);
int enable_auto_recovery(RecoveryContext* ctx, bool enabled);
void set_crash_handler(RecoveryContext* ctx, void (*handler)(void*), void* handler_ctx);

#endif 