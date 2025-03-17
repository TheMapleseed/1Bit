#include "onebit/onebit_utils.h"
#include "onebit/onebit_error.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/time.h>
#endif

// Global state for logging
static struct {
    FILE* log_file;
    LogLevel current_level;
    bool initialized;
} g_log_ctx = {NULL, LOG_INFO, false};

// File operations
int utils_read_file(const char* filename, void* buffer, size_t* size) {
    if (!filename || !buffer || !size) return ONEBIT_ERROR_INVALID_PARAM;
    
    FILE* file = fopen(filename, "rb");
    if (!file) return ONEBIT_ERROR_FILE_OPEN;
    
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    if (file_size > (long)*size) {
        fclose(file);
        return ONEBIT_ERROR_BUFFER_SIZE;
    }
    
    size_t read_size = fread(buffer, 1, file_size, file);
    fclose(file);
    
    if (read_size != (size_t)file_size) {
        return ONEBIT_ERROR_FILE_READ;
    }
    
    *size = read_size;
    return ONEBIT_SUCCESS;
}

int utils_write_file(const char* filename, const void* data, size_t size) {
    if (!filename || !data) return ONEBIT_ERROR_INVALID_PARAM;
    
    FILE* file = fopen(filename, "wb");
    if (!file) return ONEBIT_ERROR_FILE_OPEN;
    
    size_t written = fwrite(data, 1, size, file);
    fclose(file);
    
    return (written == size) ? ONEBIT_SUCCESS : ONEBIT_ERROR_FILE_WRITE;
}

bool utils_file_exists(const char* filename) {
    if (!filename) return false;
    FILE* file = fopen(filename, "rb");
    if (file) {
        fclose(file);
        return true;
    }
    return false;
}

size_t utils_file_size(const char* filename) {
    if (!filename) return 0;
    
    FILE* file = fopen(filename, "rb");
    if (!file) return 0;
    
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fclose(file);
    
    return (size < 0) ? 0 : (size_t)size;
}

// Memory operations
void* utils_aligned_alloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

void utils_aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void utils_zero_memory(void* ptr, size_t size) {
    if (ptr) memset(ptr, 0, size);
}

void utils_copy_memory(void* dst, const void* src, size_t size) {
    if (dst && src) memcpy(dst, src, size);
}

// String operations
char* utils_strdup(const char* str) {
    return str ? strdup(str) : NULL;
}

size_t utils_strlen(const char* str) {
    return str ? strlen(str) : 0;
}

int utils_strcmp(const char* str1, const char* str2) {
    if (!str1 || !str2) return str1 ? 1 : (str2 ? -1 : 0);
    return strcmp(str1, str2);
}

char* utils_strtok(char* str, const char* delim, char** saveptr) {
    return strtok_r(str, delim, saveptr);
}

// Random number generation
static uint32_t g_random_state = 0;

void utils_random_seed(uint32_t seed) {
    g_random_state = seed;
}

uint32_t utils_random_uint32(void) {
    // xoshiro128** algorithm
    g_random_state += 0x9e3779b9;
    uint32_t result = ((g_random_state >> 16) ^ g_random_state) * 0x85ebca6b;
    result = ((result >> 13) ^ result) * 0xc2b2ae35;
    return (result >> 16) ^ result;
}

float utils_random_float(void) {
    return (float)utils_random_uint32() / (float)UINT32_MAX;
}

float utils_random_normal(float mean, float stddev) {
    // Box-Muller transform
    float u1 = utils_random_float();
    float u2 = utils_random_float();
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    return mean + stddev * z;
}

// Time operations
uint64_t utils_time_us(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (uint64_t)(count.QuadPart * 1000000LL / freq.QuadPart);
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
#endif
}

void utils_sleep_ms(uint32_t milliseconds) {
#ifdef _WIN32
    Sleep(milliseconds);
#else
    usleep(milliseconds * 1000);
#endif
}

// Logging utilities
void utils_log_init(const char* filename) {
    if (g_log_ctx.initialized) utils_log_close();
    
    if (filename) {
        g_log_ctx.log_file = fopen(filename, "a");
    }
    g_log_ctx.initialized = true;
}

void utils_log_close(void) {
    if (g_log_ctx.log_file) {
        fclose(g_log_ctx.log_file);
        g_log_ctx.log_file = NULL;
    }
    g_log_ctx.initialized = false;
}

void utils_log(LogLevel level, const char* format, ...) {
    if (level < g_log_ctx.current_level) return;
    
    static const char* level_str[] = {
        "DEBUG", "INFO", "WARNING", "ERROR"
    };
    
    time_t now;
    time(&now);
    char time_str[32];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&now));
    
    va_list args;
    va_start(args, format);
    
    if (g_log_ctx.log_file) {
        fprintf(g_log_ctx.log_file, "[%s] %s: ", time_str, level_str[level]);
        vfprintf(g_log_ctx.log_file, format, args);
        fprintf(g_log_ctx.log_file, "\n");
        fflush(g_log_ctx.log_file);
    }
    
    va_end(args);
}

void utils_set_log_level(LogLevel level) {
    g_log_ctx.current_level = level;
}

// Math utilities
float utils_rsqrt(float x) {
    // Fast inverse square root approximation
    union {
        float f;
        uint32_t i;
    } conv = { .f = x };
    conv.i = 0x5f3759df - (conv.i >> 1);
    conv.f *= 1.5f - (x * 0.5f * conv.f * conv.f);
    return conv.f;
}

float utils_fast_exp(float x) {
    // Fast exponential approximation
    x = 1.0f + x / 256.0f;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x; x *= x; x *= x;
    return x;
}

float utils_fast_tanh(float x) {
    // Fast tanh approximation
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return a / b;
}

void utils_softmax(float* x, size_t n) {
    if (!x || n == 0) return;
    
    float max_val = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        x[i] = utils_fast_exp(x[i] - max_val);
        sum += x[i];
    }
    
    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}

// Bit manipulation
uint32_t utils_popcount(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    x = x + (x >> 8);
    x = x + (x >> 16);
    return x & 0x3F;
}

uint32_t utils_clz(uint32_t x) {
    if (x == 0) return 32;
    
    uint32_t n = 0;
    if ((x & 0xFFFF0000) == 0) { n += 16; x <<= 16; }
    if ((x & 0xFF000000) == 0) { n += 8;  x <<= 8;  }
    if ((x & 0xF0000000) == 0) { n += 4;  x <<= 4;  }
    if ((x & 0xC0000000) == 0) { n += 2;  x <<= 2;  }
    if ((x & 0x80000000) == 0) { n += 1;  x <<= 1;  }
    return n;
}

uint32_t utils_ctz(uint32_t x) {
    if (x == 0) return 32;
    
    uint32_t n = 0;
    if ((x & 0x0000FFFF) == 0) { n += 16; x >>= 16; }
    if ((x & 0x000000FF) == 0) { n += 8;  x >>= 8;  }
    if ((x & 0x0000000F) == 0) { n += 4;  x >>= 4;  }
    if ((x & 0x00000003) == 0) { n += 2;  x >>= 2;  }
    if ((x & 0x00000001) == 0) { n += 1;  x >>= 1;  }
    return n;
}

uint32_t utils_rotl(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

uint32_t utils_rotr(uint32_t x, int k) {
    return (x >> k) | (x << (32 - k));
} 