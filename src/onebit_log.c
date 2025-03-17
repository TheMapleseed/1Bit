#include "onebit/onebit_log.h"
#include "onebit/onebit_error.h"
#include <stdarg.h>
#include <time.h>

static const char* level_strings[] = {
    "TRACE",
    "DEBUG",
    "INFO",
    "WARN",
    "ERROR",
    "FATAL"
};

static const char* level_colors[] = {
    "\x1b[94m", // Trace - Light blue
    "\x1b[36m", // Debug - Cyan
    "\x1b[32m", // Info - Green
    "\x1b[33m", // Warn - Yellow
    "\x1b[31m", // Error - Red
    "\x1b[35m"  // Fatal - Magenta
};

int log_init(LogContext* ctx, const LogConfig* config) {
    if (!ctx) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->level = config ? config->level : LOG_LEVEL_INFO;
    ctx->use_colors = config ? config->use_colors : true;
    ctx->file = NULL;
    
    if (config && config->filename) {
        ctx->file = fopen(config->filename, "a");
        if (!ctx->file) {
            return ONEBIT_ERROR_IO;
        }
    }
    
    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        if (ctx->file) fclose(ctx->file);
        return ONEBIT_ERROR_THREAD;
    }
    
    return ONEBIT_SUCCESS;
}

void log_cleanup(LogContext* ctx) {
    if (!ctx) return;
    
    if (ctx->file) {
        fclose(ctx->file);
    }
    
    pthread_mutex_destroy(&ctx->mutex);
}

static void log_write(LogContext* ctx, LogLevel level,
                     const char* file, int line,
                     const char* fmt, va_list args) {
    if (level < ctx->level) return;
    
    time_t t = time(NULL);
    struct tm* lt = localtime(&t);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", lt);
    
    pthread_mutex_lock(&ctx->mutex);
    
    // Write to console
    if (ctx->use_colors) {
        fprintf(stderr, "%s", level_colors[level]);
    }
    
    fprintf(stderr, "[%s] %-5s ", timestamp, level_strings[level]);
    
    if (file) {
        fprintf(stderr, "%s:%d: ", file, line);
    }
    
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    
    if (ctx->use_colors) {
        fprintf(stderr, "\x1b[0m");
    }
    
    // Write to file if enabled
    if (ctx->file) {
        fprintf(ctx->file, "[%s] %-5s ", timestamp,
                level_strings[level]);
        
        if (file) {
            fprintf(ctx->file, "%s:%d: ", file, line);
        }
        
        vfprintf(ctx->file, fmt, args);
        fprintf(ctx->file, "\n");
        fflush(ctx->file);
    }
    
    pthread_mutex_unlock(&ctx->mutex);
}

void log_trace(LogContext* ctx, const char* file, int line,
               const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_write(ctx, LOG_LEVEL_TRACE, file, line, fmt, args);
    va_end(args);
}

void log_debug(LogContext* ctx, const char* file, int line,
               const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_write(ctx, LOG_LEVEL_DEBUG, file, line, fmt, args);
    va_end(args);
}

void log_info(LogContext* ctx, const char* file, int line,
               const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_write(ctx, LOG_LEVEL_INFO, file, line, fmt, args);
    va_end(args);
}

void log_warn(LogContext* ctx, const char* file, int line,
               const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_write(ctx, LOG_LEVEL_WARN, file, line, fmt, args);
    va_end(args);
}

void log_error(LogContext* ctx, const char* file, int line,
               const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_write(ctx, LOG_LEVEL_ERROR, file, line, fmt, args);
    va_end(args);
}

void log_fatal(LogContext* ctx, const char* file, int line,
               const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_write(ctx, LOG_LEVEL_FATAL, file, line, fmt, args);
    va_end(args);
} 