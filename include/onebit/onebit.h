/**
 * @file onebit.h
 * @brief Main header file for OneBit neural network inference library
 *
 * OneBit is a high-performance library for efficient neural network inference
 * with binary weight quantization support. It provides optimized implementations
 * for x86 (AVX-512, AVX2, SSE4) and GPU acceleration (CUDA, ROCm).
 *
 * @author OneBit Team
 * @version 1.0.0
 */

#ifndef ONEBIT_onebit_H
#define ONEBIT_onebit_H

#ifdef __cplusplus
extern "C" {
#endif

// Version information
#define ONEBIT_VERSION_MAJOR 1
#define ONEBIT_VERSION_MINOR 0
#define ONEBIT_VERSION_PATCH 0
#define ONEBIT_VERSION "1.0.0"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Error codes
#define ONEBIT_SUCCESS               0
#define ONEBIT_ERROR_INVALID_PARAM  -1
#define ONEBIT_ERROR_MEMORY         -2
#define ONEBIT_ERROR_IO             -3
#define ONEBIT_ERROR_RUNTIME        -4
#define ONEBIT_ERROR_THREAD         -5
#define ONEBIT_ERROR_NOT_SUPPORTED  -6
#define ONEBIT_ERROR_CUDA           -7
#define ONEBIT_ERROR_METAL          -8
#define ONEBIT_ERROR_ROCM           -9

// Model parameter types
typedef enum {
    PARAM_TYPE_FLOAT32,
    PARAM_TYPE_FLOAT16,
    PARAM_TYPE_INT8,
    PARAM_TYPE_INT4,
    PARAM_TYPE_BINARY
} ParamType;

// Configuration structure
typedef struct {
    // Model configuration
    int vocab_size;
    int hidden_size;
    int num_layers;
    int num_heads;
    int intermediate_size;
    int max_position;
    
    // Training configuration
    float attention_dropout;
    float hidden_dropout;
    float embedding_dropout;
    float layer_norm_eps;
    
    // Runtime configuration
    bool use_cuda;
    bool use_rocm;
    bool use_metal;
    int device_id;
    bool use_quantization;
    int num_threads;
    
    // Memory configuration
    size_t memory_pool_size;
    bool use_memory_pool;
    
    // Tokenizer configuration
    char* vocab_file;
    bool add_special_tokens;
} OneBitConfig;

// Forward declarations for opaque structures
typedef struct OneBitModelStruct OneBitModel;
typedef struct OneBitTokenizerStruct OneBitTokenizer;
typedef struct OneBitMemoryPoolStruct OneBitMemoryPool;
typedef struct OneBitComputeContextStruct OneBitComputeContext;

// OneBit context
typedef struct {
    OneBitModel* model;
    OneBitTokenizer* tokenizer;
    OneBitMemoryPool* pool;
    OneBitComputeContext* compute;
    OneBitConfig config;
    bool is_initialized;
} OneBitContext;

// Function declarations
int onebit_init(OneBitContext* ctx, const OneBitConfig* config);
int onebit_load_model(OneBitContext* ctx, const char* model_path);
int onebit_generate(OneBitContext* ctx, const char* prompt,
                   char* output, int max_length,
                   float temperature, float top_p);
void onebit_cleanup(OneBitContext* ctx);

// Helper functions
const char* onebit_version(void);
const char* onebit_error_string(int error_code);
void onebit_print_config(const OneBitConfig* config);
OneBitConfig onebit_default_config(void);

// Device information
int onebit_get_device_count(void);
int onebit_get_device_info(int device_id, char* name, size_t* memory);
bool onebit_has_avx512(void);
bool onebit_has_avx2(void);
bool onebit_has_sse4(void);

#ifdef __cplusplus
}
#endif

#endif /* ONEBIT_H */ 