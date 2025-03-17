#ifndef ONEBIT_H
#define ONEBIT_H

#ifdef __cplusplus
extern "C" {
#endif

// Include all component headers
#include "onebit_model.h"
#include "onebit_tokenizer.h"
#include "onebit_error.h"
#include "onebit_memory.h"

// Version information
#define ONEBIT_VERSION_MAJOR 1
#define ONEBIT_VERSION_MINOR 0
#define ONEBIT_VERSION_PATCH 0

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

// OneBit context
typedef struct {
    OneBitModel* model;
    Tokenizer* tokenizer;
    OneBitMemoryPool* pool;
    OneBitCUDAContext* cuda;
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
void onebit_print_config(const OneBitConfig* config);
OneBitConfig onebit_default_config(void);

#ifdef __cplusplus
}
#endif 