#include "onebit/onebit_time.h"
#include <string.h>
#include <stdio.h>

// Conversion factors
static const uint64_t NANOS_PER_SEC = 1000000000ULL;
static const uint64_t NANOS_PER_MSEC = 1000000ULL;
static const uint64_t NANOS_PER_USEC = 1000ULL;

static uint64_t timespec_to_nanos(const struct timespec* ts) {
    return ts->tv_sec * NANOS_PER_SEC + ts->tv_nsec;
}

static void nanos_to_timespec(uint64_t nanos, struct timespec* ts) {
    ts->tv_sec = nanos / NANOS_PER_SEC;
    ts->tv_nsec = nanos % NANOS_PER_SEC;
}

static double nanos_to_unit(uint64_t nanos, TimeUnit unit) {
    switch (unit) {
        case TIME_UNIT_NANOSECONDS:
            return (double)nanos;
        case TIME_UNIT_MICROSECONDS:
            return (double)nanos / NANOS_PER_USEC;
        case TIME_UNIT_MILLISECONDS:
            return (double)nanos / NANOS_PER_MSEC;
        case TIME_UNIT_SECONDS:
            return (double)nanos / NANOS_PER_SEC;
        default:
            return 0.0;
    }
}

void timer_init(TimerContext* ctx) {
    if (!ctx) return;
    
    memset(ctx, 0, sizeof(TimerContext));
    ctx->running = false;
}

void timer_start(TimerContext* ctx) {
    if (!ctx) return;
    
    clock_gettime(CLOCK_MONOTONIC, &ctx->start_time);
    ctx->running = true;
}

void timer_stop(TimerContext* ctx) {
    if (!ctx || !ctx->running) return;
    
    clock_gettime(CLOCK_MONOTONIC, &ctx->end_time);
    ctx->running = false;
}

void timer_reset(TimerContext* ctx) {
    if (!ctx) return;
    
    memset(&ctx->start_time, 0, sizeof(struct timespec));
    memset(&ctx->end_time, 0, sizeof(struct timespec));
    ctx->running = false;
}

double timer_elapsed(const TimerContext* ctx, TimeUnit unit) {
    if (!ctx) return 0.0;
    
    struct timespec current_time;
    const struct timespec* end_time;
    
    if (ctx->running) {
        clock_gettime(CLOCK_MONOTONIC, &current_time);
        end_time = &current_time;
    } else {
        end_time = &ctx->end_time;
    }
    
    uint64_t start_nanos = timespec_to_nanos(&ctx->start_time);
    uint64_t end_nanos = timespec_to_nanos(end_time);
    
    return nanos_to_unit(end_nanos - start_nanos, unit);
}

void time_sleep(double duration, TimeUnit unit) {
    uint64_t nanos;
    
    switch (unit) {
        case TIME_UNIT_NANOSECONDS:
            nanos = (uint64_t)duration;
            break;
        case TIME_UNIT_MICROSECONDS:
            nanos = (uint64_t)(duration * NANOS_PER_USEC);
            break;
        case TIME_UNIT_MILLISECONDS:
            nanos = (uint64_t)(duration * NANOS_PER_MSEC);
            break;
        case TIME_UNIT_SECONDS:
            nanos = (uint64_t)(duration * NANOS_PER_SEC);
            break;
        default:
            return;
    }
    
    struct timespec ts;
    nanos_to_timespec(nanos, &ts);
    nanosleep(&ts, NULL);
}

uint64_t time_now(TimeUnit unit) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    
    uint64_t nanos = timespec_to_nanos(&ts);
    return (uint64_t)nanos_to_unit(nanos, unit);
}

double time_convert(double time, TimeUnit from, TimeUnit to) {
    // Convert to nanoseconds first
    uint64_t nanos;
    
    switch (from) {
        case TIME_UNIT_NANOSECONDS:
            nanos = (uint64_t)time;
            break;
        case TIME_UNIT_MICROSECONDS:
            nanos = (uint64_t)(time * NANOS_PER_USEC);
            break;
        case TIME_UNIT_MILLISECONDS:
            nanos = (uint64_t)(time * NANOS_PER_MSEC);
            break;
        case TIME_UNIT_SECONDS:
            nanos = (uint64_t)(time * NANOS_PER_SEC);
            break;
        default:
            return 0.0;
    }
    
    return nanos_to_unit(nanos, to);
}

size_t time_format(char* buffer, size_t size, const char* format) {
    if (!buffer || !format || size == 0) return 0;
    
    time_t now;
    struct tm tm_info;
    
    time(&now);
    localtime_r(&now, &tm_info);
    
    return strftime(buffer, size, format, &tm_info);
}

bool time_parse(const char* str, const char* format, time_t* timestamp) {
    if (!str || !format || !timestamp) return false;
    
    struct tm tm_info = {0};
    char* result = strptime(str, format, &tm_info);
    
    if (!result || *result != '\0') return false;
    
    *timestamp = mktime(&tm_info);
    return *timestamp != -1;
} 