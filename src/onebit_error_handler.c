#include "onebit/onebit_error_handler.h"
#include <string.h>
#include <stdio.h>

// Thread-local error state
static __thread struct {
    OneBitError code;
    char message[ERROR_MAX_LENGTH];
    const char* file;
    int line;
} error_state = {ONEBIT_SUCCESS};

// Error code messages
static const char* error_messages[] = {
    [ONEBIT_SUCCESS] = "Success",
    [ONEBIT_ERROR_DEVICE_NOT_FOUND] = "No compatible compute device found",
    [ONEBIT_ERROR_CUDA_INIT] = "CUDA initialization failed",
    [ONEBIT_ERROR_METAL_INIT] = "Metal initialization failed",
    [ONEBIT_ERROR_CPU_INIT] = "CPU initialization failed",
    [ONEBIT_ERROR_MEMORY] = "Memory allocation failed",
    [ONEBIT_ERROR_INVALID_DEVICE] = "Invalid device specified",
    [ONEBIT_ERROR_DEVICE_LOST] = "Device connection lost",
    [ONEBIT_ERROR_INITIALIZATION] = "Initialization error",
    [ONEBIT_ERROR_ALREADY_INITIALIZED] = "Already initialized"
};

void onebit_set_error(OneBitError code, const char* file, int line) {
    error_state.code = code;
    error_state.file = file;
    error_state.line = line;
    strncpy(error_state.message, error_messages[code], ERROR_MAX_LENGTH - 1);
}

void onebit_set_error_msg(OneBitError code, const char* msg, const char* file, int line) {
    error_state.code = code;
    error_state.file = file;
    error_state.line = line;
    strncpy(error_state.message, msg, ERROR_MAX_LENGTH - 1);
}

OneBitError onebit_get_error(void) {
    return error_state.code;
}

const char* onebit_get_error_message(void) {
    return error_state.message;
}

void onebit_clear_error(void) {
    error_state.code = ONEBIT_SUCCESS;
    error_state.message[0] = '\0';
    error_state.file = NULL;
    error_state.line = 0;
}

const char* onebit_get_error_location(void) {
    static char location[ERROR_MAX_LENGTH];
    if (error_state.file) {
        snprintf(location, ERROR_MAX_LENGTH, "%s:%d", 
                error_state.file, error_state.line);
        return location;
    }
    return "unknown location";
} 