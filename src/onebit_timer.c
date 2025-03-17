#include "onebit/onebit_timer.h"
#include "onebit/onebit_error.h"
#include <string.h>
#include <time.h>

// Timer entry
typedef struct TimerEntry {
    char* name;
    uint64_t start_time;
    uint64_t total_time;
    uint64_t call_count;
    bool running;
    struct TimerEntry* next;
} TimerEntry;

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int timer_init(TimerContext* ctx, const TimerConfig* config) {
    if (!ctx) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->timers = NULL;
    ctx->num_timers = 0;
    
    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        return ONEBIT_ERROR_THREAD;
    }
    
    return ONEBIT_SUCCESS;
}

void timer_cleanup(TimerContext* ctx) {
    if (!ctx) return;
    
    pthread_mutex_lock(&ctx->mutex);
    
    TimerEntry* timer = ctx->timers;
    while (timer) {
        TimerEntry* next = timer->next;
        free(timer->name);
        free(timer);
        timer = next;
    }
    
    pthread_mutex_unlock(&ctx->mutex);
    pthread_mutex_destroy(&ctx->mutex);
}

static TimerEntry* find_timer(TimerContext* ctx, const char* name) {
    TimerEntry* timer = ctx->timers;
    while (timer) {
        if (strcmp(timer->name, name) == 0) {
            return timer;
        }
        timer = timer->next;
    }
    return NULL;
} 