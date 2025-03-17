/**
 * @file onebit_error.h
 * @brief Error handling for OneBit
 *
 * Provides error handling, reporting, and logging functionality.
 *
 * @author OneBit Team
 * @version 1.0.0
 */

#ifndef ONEBIT_onebit_error_H
#define ONEBIT_onebit_error_H

#include <stdbool.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error handling functions
void error_set(int error_code, const char* format, ...);
const char* error_get_message(void);
int error_get_code(void);
void error_clear(void);

// Error callback
void error_set_callback(void (*callback)(int error_code, const char* error_message));
void error_default_handler(int error_code, const char* error_message);

// Error system management
int error_init(void);
void error_cleanup(void);

// Error utilities
const char* error_code_to_string(int error_code);
bool error_check(int error_code, const char* file, int line, const char* func);
void error_log(int error_code, const char* file, int line, const char* func, const char* format, ...);
bool error_assert(bool condition, int error_code, const char* file, int line, const char* func, const char* message);

// Convenience macros for error handling
#define CHECK_ERROR(call) error_check((call), __FILE__, __LINE__, __func__)
#define ASSERT_ERROR(condition, error_code, message) error_assert((condition), (error_code), __FILE__, __LINE__, __func__, (message))
#define LOG_ERROR(error_code, format, ...) error_log((error_code), __FILE__, __LINE__, __func__, (format), ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif /* ONEBIT_ERROR_H */ 