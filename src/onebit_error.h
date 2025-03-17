#ifndef ONEBIT_ERROR_H
#define ONEBIT_ERROR_H

// Error codes
#define ONEBIT_SUCCESS 0
#define ONEBIT_ERROR_MEMORY -1
#define ONEBIT_ERROR_IO -2
#define ONEBIT_ERROR_INVALID_PARAM -3
#define ONEBIT_ERROR_INITIALIZATION -4
#define ONEBIT_ERROR_NOT_IMPLEMENTED -5
#define ONEBIT_ERROR_CUDA -6
#define ONEBIT_ERROR_OVERFLOW -7

// Error checking macros
#define ONEBIT_CHECK_ERROR(condition) \
    do { \
        if (!(condition)) { \
            return ONEBIT_ERROR_INVALID_PARAM; \
        } \
    } while (0)

#define ONEBIT_CHECK_NULL(ptr) \
    do { \
        if (!(ptr)) { \
            return ONEBIT_ERROR_MEMORY; \
        } \
    } while (0)

// Function declarations
const char* onebit_error_string(int error_code);
void onebit_set_error_callback(void (*callback)(int, const char*));
void onebit_clear_error(void);

#endif 