#include "onebit/onebit_profiler.h"
#include "onebit/onebit_error.h"
#include <string.h>
#include <time.h>

// Internal profiling event
typedef struct {
    char name[64];
    uint64_t start_time;
    uint64_t end_time;
    size_t memory_usage;
    ProfileEventType type;
} ProfileEvent;

static uint64_t get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

int profiler_init(ProfilerContext* ctx, const ProfilerConfig* config) {
    if (!ctx || !config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->events = malloc(config->max_events * sizeof(ProfileEvent));
    if (!ctx->events) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    ctx->num_events = 0;
    ctx->max_events = config->max_events;
    ctx->enabled = config->enabled;
    ctx->output_file = NULL;
    
    if (config->output_file) {
        ctx->output_file = fopen(config->output_file, "w");
        if (!ctx->output_file) {
            free(ctx->events);
            return ONEBIT_ERROR_IO;
        }
        
        // Write header
        fprintf(ctx->output_file,
                "name,type,start_time,end_time,duration_us,memory_bytes\n");
    }
    
    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        if (ctx->output_file) fclose(ctx->output_file);
        free(ctx->events);
        return ONEBIT_ERROR_THREAD;
    }
    
    return ONEBIT_SUCCESS;
}

void profiler_cleanup(ProfilerContext* ctx) {
    if (!ctx) return;
    
    pthread_mutex_lock(&ctx->mutex);
    
    free(ctx->events);
    
    if (ctx->output_file) {
        fclose(ctx->output_file);
    }
    
    pthread_mutex_unlock(&ctx->mutex);
    pthread_mutex_destroy(&ctx->mutex);
}

int profiler_start_event(ProfilerContext* ctx, const char* name,
                        ProfileEventType type) {
    if (!ctx || !name) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    if (!ctx->enabled) {
        return ONEBIT_SUCCESS;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    if (ctx->num_events >= ctx->max_events) {
        pthread_mutex_unlock(&ctx->mutex);
        return ONEBIT_ERROR_OVERFLOW;
    }
    
    ProfileEvent* event = &ctx->events[ctx->num_events++];
    strncpy(event->name, name, sizeof(event->name) - 1);
    event->name[sizeof(event->name) - 1] = '\0';
    event->start_time = get_time_us();
    event->end_time = 0;
    event->memory_usage = 0;
    event->type = type;
    
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
}

int profiler_end_event(ProfilerContext* ctx, const char* name,
                      size_t memory_usage) {
    if (!ctx || !name) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    if (!ctx->enabled) {
        return ONEBIT_SUCCESS;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    // Find matching start event
    ProfileEvent* event = NULL;
    for (int i = ctx->num_events - 1; i >= 0; i--) {
        if (strcmp(ctx->events[i].name, name) == 0 &&
            ctx->events[i].end_time == 0) {
            event = &ctx->events[i];
            break;
        }
    }
    
    if (!event) {
        pthread_mutex_unlock(&ctx->mutex);
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    event->end_time = get_time_us();
    event->memory_usage = memory_usage;
    
    // Write to output file if enabled
    if (ctx->output_file) {
        fprintf(ctx->output_file, "%s,%d,%lu,%lu,%lu,%zu\n",
                event->name, event->type, event->start_time, event->end_time,
                event->end_time - event->start_time, event->memory_usage);
    }
    
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
} 