#include "onebit/onebit_logging.h"
#include "onebit/onebit_error.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

// Internal logging state
static LogLevel current_log_level = LOG_INFO;
static FILE* log_file = NULL;
static LogCallback log_callback = NULL;
static void* callback_ctx = NULL;

static const char* level_strings[] = {
    "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"
};

static const char* level_colors[] = {
    "\x1b[90m", // Gray for TRACE
    "\x1b[36m", // Cyan for DEBUG
    "\x1b[32m", // Green for INFO
    "\x1b[33m", // Yellow for WARN
    "\x1b[31m", // Red for ERROR
    "\x1b[35m"  // Magenta for FATAL
};

int logging_init(const LogConfig* config) {
    if (!config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    current_log_level = config->level;
    
    if (config->filename) {
        log_file = fopen(config->filename, "a");
        if (!log_file) {
            return ONEBIT_ERROR_IO;
        }
    }
    
    log_callback = config->callback;
    callback_ctx = config->callback_ctx;
    
    return ONEBIT_SUCCESS;
}

void logging_cleanup(void) {
    if (log_file) {
        fclose(log_file);
        log_file = NULL;
    }
}

void logging_set_level(LogLevel level) {
    current_log_level = level;
}

LogLevel logging_get_level(void) {
    return current_log_level;
}

void logging_set_callback(LogCallback callback, void* ctx) {
    log_callback = callback;
    callback_ctx = ctx;
}

static void format_timestamp(char* buffer, size_t size) {
    time_t now;
    struct tm* timeinfo;
    
    time(&now);
    timeinfo = localtime(&now);
    
    strftime(buffer, size, "%Y-%m-%d %H:%M:%S", timeinfo);
}

void logging_log(LogLevel level, const char* file, int line,
                const char* fmt, ...) {
    if (level < current_log_level) {
        return;
    }
    
    char timestamp[32];
    format_timestamp(timestamp, sizeof(timestamp));
    
    char message[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);
    
    // Remove trailing newlines
    size_t len = strlen(message);
    while (len > 0 && (message[len-1] == '\n' || message[len-1] == '\r')) {
        message[--len] = '\0';
    }
    
    // Format full log entry
    char entry[2048];
    snprintf(entry, sizeof(entry), "%s [%s] %s:%d - %s\n",
             timestamp, level_strings[level], file, line, message);
    
    // Write to console with colors
    if (level >= current_log_level) {
        fprintf(stderr, "%s%s\x1b[0m", level_colors[level], entry);
    }
    
    // Write to log file
    if (log_file) {
        fputs(entry, log_file);
        fflush(log_file);
    }
    
    // Call user callback if set
    if (log_callback) {
        log_callback(level, file, line, message, callback_ctx);
    }
}

void logging_flush(void) {
    if (log_file) {
        fflush(log_file);
    }
}

const char* logging_level_string(LogLevel level) {
    if (level < 0 || level >= sizeof(level_strings)/sizeof(char*)) {
        return "UNKNOWN";
    }
    return level_strings[level];
}

bool logging_would_log(LogLevel level) {
    return level >= current_log_level;
}

int logging_set_file(const char* filename) {
    if (log_file) {
        fclose(log_file);
    }
    
    if (filename) {
        log_file = fopen(filename, "a");
        if (!log_file) {
            return ONEBIT_ERROR_IO;
        }
    } else {
        log_file = NULL;
    }
    
    return ONEBIT_SUCCESS;
} 