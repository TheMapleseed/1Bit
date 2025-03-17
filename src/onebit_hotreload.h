#ifndef ONEBIT_HOTRELOAD_H
#define ONEBIT_HOTRELOAD_H

#include <stdbool.h>

// Hot reload configuration
typedef struct {
    char watch_dir[256];
    char* file_patterns[16];
    int num_patterns;
    int poll_interval_ms;
    bool reload_on_change;
    void (*reload_callback)(const char* file, void* ctx);
    void* callback_ctx;
} HotReloadConfig;

// Hot reload context
typedef struct {
    HotReloadConfig config;
    void* file_monitor;
    bool is_running;
    pthread_t monitor_thread;
    pthread_mutex_t reload_mutex;
} HotReloadContext;

// Function declarations
int hotreload_init(HotReloadContext* ctx, const HotReloadConfig* config);
void hotreload_cleanup(HotReloadContext* ctx);
int hotreload_start(HotReloadContext* ctx);
int hotreload_stop(HotReloadContext* ctx);
int reload_model(OneBitContext* ctx, const char* model_path);

#endif 