#include <onebit/onebit.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

// Thread-local error information
static __thread char error_buffer[1024];
static __thread int last_error_code = ONEBIT_SUCCESS;

// Mutex for global error handling
static pthread_mutex_t error_mutex = PTHREAD_MUTEX_INITIALIZER;

// Global error callback
static void (*global_error_callback)(int error_code, const char* error_message) = NULL;

// Set the error message and code
void error_set(int error_code, const char* format, ...) {
    va_list args;
    va_start(args, format);
    
    // Format the error message
    vsnprintf(error_buffer, sizeof(error_buffer), format, args);
    last_error_code = error_code;
    
    va_end(args);
    
    // Call the global error callback if set
    if (global_error_callback) {
        pthread_mutex_lock(&error_mutex);
        global_error_callback(error_code, error_buffer);
        pthread_mutex_unlock(&error_mutex);
    }
}

// Get the last error message
const char* error_get_message(void) {
    return error_buffer;
}

// Get the last error code
int error_get_code(void) {
    return last_error_code;
}

// Clear the error state
void error_clear(void) {
    error_buffer[0] = '\0';
    last_error_code = ONEBIT_SUCCESS;
}

// Set the global error callback
void error_set_callback(void (*callback)(int error_code, const char* error_message)) {
    pthread_mutex_lock(&error_mutex);
    global_error_callback = callback;
    pthread_mutex_unlock(&error_mutex);
}

// Default error handler that prints to stderr
void error_default_handler(int error_code, const char* error_message) {
    fprintf(stderr, "OneBit Error [%d]: %s\n", error_code, error_message);
}

// Initialize the error handling system
int error_init(void) {
    // Set the default error callback
    error_set_callback(error_default_handler);
    return ONEBIT_SUCCESS;
}

// Cleanup the error handling system
void error_cleanup(void) {
    pthread_mutex_lock(&error_mutex);
    global_error_callback = NULL;
    pthread_mutex_unlock(&error_mutex);
}

// Convert error code to string
const char* error_code_to_string(int error_code) {
    switch (error_code) {
        case ONEBIT_SUCCESS:
            return "Success";
        case ONEBIT_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case ONEBIT_ERROR_MEMORY:
            return "Memory allocation error";
        case ONEBIT_ERROR_IO:
            return "I/O error";
        case ONEBIT_ERROR_RUNTIME:
            return "Runtime error";
        case ONEBIT_ERROR_THREAD:
            return "Threading error";
        case ONEBIT_ERROR_NOT_SUPPORTED:
            return "Operation not supported";
        case ONEBIT_ERROR_CUDA:
            return "CUDA error";
        case ONEBIT_ERROR_METAL:
            return "Metal error";
        case ONEBIT_ERROR_ROCM:
            return "ROCm error";
        default:
            return "Unknown error";
    }
}

// Check if an operation succeeded and set error if not
bool error_check(int error_code, const char* file, int line, const char* func) {
    if (error_code != ONEBIT_SUCCESS) {
        error_set(error_code, "Error %d (%s) at %s:%d in function %s",
                 error_code, error_code_to_string(error_code), file, line, func);
        return false;
    }
    return true;
}

// Log an error message
void error_log(int error_code, const char* file, int line, const char* func, const char* format, ...) {
    va_list args;
    va_start(args, format);
    
    // Format the message
    char message[1024];
    vsnprintf(message, sizeof(message), format, args);
    
    va_end(args);
    
    // Set the error
    error_set(error_code, "%s (at %s:%d in function %s)", message, file, line, func);
}

// Assert a condition and set error if it fails
bool error_assert(bool condition, int error_code, const char* file, int line, const char* func, const char* message) {
    if (!condition) {
        error_set(error_code, "Assertion failed: %s (at %s:%d in function %s)",
                 message, file, line, func);
        return false;
    }
    return true;
}

// Convenience macros for common error handling patterns
#define CHECK_ERROR(call) error_check((call), __FILE__, __LINE__, __func__)
#define ASSERT_ERROR(condition, error_code, message) error_assert((condition), (error_code), __FILE__, __LINE__, __func__, (message))
#define LOG_ERROR(error_code, format, ...) error_log((error_code), __FILE__, __LINE__, __func__, (format), ##__VA_ARGS__)

// Error message lookup table
static const char* error_messages[] = {
    "Success",
    "Invalid parameter",
    "Memory allocation failed",
    "I/O error",
    "Thread error",
    "Device error",
    "Not implemented",
    "Invalid format",
    "Buffer overflow",
    "Operation timeout",
    "Operation cancelled",
    "Resource not found",
    "Resource already exists",
    "Permission denied",
    "Network error",
    "Protocol error",
    "Busy",
    "Temporary failure",
    "Invalid state",
    "Quota exceeded",
    "Range error",
    "Data error",
    "Unknown error"
};

// Thread-local error state
static __thread struct {
    int code;
    char message[256];
} error_state = {ONEBIT_SUCCESS, ""};

const char* error_string(int error_code) {
    if (error_code < 0 || 
        error_code >= sizeof(error_messages)/sizeof(char*)) {
        return error_messages[ONEBIT_ERROR_UNKNOWN];
    }
    return error_messages[error_code];
}

void error_set(int error_code, const char* format, ...) {
    error_state.code = error_code;
    
    va_list args;
    va_start(args, format);
    vsnprintf(error_state.message, sizeof(error_state.message),
              format, args);
    va_end(args);
}

int error_get(void) {
    return error_state.code;
}

const char* error_message(void) {
    if (error_state.message[0] != '\0') {
        return error_state.message;
    }
    return error_string(error_state.code);
}

void error_clear(void) {
    error_state.code = ONEBIT_SUCCESS;
    error_state.message[0] = '\0';
}

bool error_is_error(int error_code) {
    return error_code != ONEBIT_SUCCESS;
}

int error_convert(int system_error) {
    switch (system_error) {
        case 0:
            return ONEBIT_SUCCESS;
            
        case ENOMEM:
            return ONEBIT_ERROR_MEMORY;
            
        case EINVAL:
            return ONEBIT_ERROR_INVALID_PARAM;
            
        case EIO:
            return ONEBIT_ERROR_IO;
            
        case EACCES:
        case EPERM:
            return ONEBIT_ERROR_PERMISSION;
            
        case ENOENT:
            return ONEBIT_ERROR_NOT_FOUND;
            
        case EEXIST:
            return ONEBIT_ERROR_ALREADY_EXISTS;
            
        case EAGAIN:
        case EBUSY:
            return ONEBIT_ERROR_BUSY;
            
        case ETIMEDOUT:
            return ONEBIT_ERROR_TIMEOUT;
            
        case ERANGE:
            return ONEBIT_ERROR_RANGE;
            
        case ENOSYS:
            return ONEBIT_ERROR_NOT_IMPLEMENTED;
            
        case EPROTO:
            return ONEBIT_ERROR_PROTOCOL;
            
        case EOVERFLOW:
            return ONEBIT_ERROR_OVERFLOW;
            
        case ECANCELED:
            return ONEBIT_ERROR_CANCELLED;
            
        case ENETDOWN:
        case ENETUNREACH:
        case ECONNREFUSED:
            return ONEBIT_ERROR_NETWORK;
            
        default:
            return ONEBIT_ERROR_UNKNOWN;
    }
}

void error_log(int error_code, const char* file, int line) {
    if (error_code != ONEBIT_SUCCESS) {
        const char* msg = error_message();
        fprintf(stderr, "Error at %s:%d: %s (%d)\n",
                file, line, msg, error_code);
    }
}

bool error_is_fatal(int error_code) {
    switch (error_code) {
        case ONEBIT_ERROR_MEMORY:
        case ONEBIT_ERROR_DEVICE:
        case ONEBIT_ERROR_
} 