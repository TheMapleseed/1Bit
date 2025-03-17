#ifndef ONEBIT_CORE_H
#define ONEBIT_CORE_H

#include <stdint.h>
#include <stdbool.h>

// Version information
#define ONEBIT_VERSION_MAJOR 1
#define ONEBIT_VERSION_MINOR 0
#define ONEBIT_VERSION_PATCH 0

// Maximum dimensions
#define MAX_BATCH_SIZE 512
#define MAX_SEQUENCE_LENGTH 2048
#define MAX_VOCAB_SIZE 65536
#define MAX_MODEL_LAYERS 128

// Precision options
typedef enum {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_INT8,
    PRECISION_MIXED
} PrecisionType;

// Core context structure
typedef struct {
    // Model architecture
    int num_layers;
    int hidden_dim;
    int num_heads;
    int vocab_size;
    
    // Runtime state
    void* model_state;
    void* cache_state;
    void* compute_state;
    
    // Configuration
    PrecisionType precision;
    bool use_cache;
    int device_id;
    
    // Statistics
    uint64_t total_tokens;
    double compute_time;
    size_t memory_used;
} OneBitContext;

// Function declarations
int onebit_init_context(OneBitContext* ctx);
void onebit_cleanup_context(OneBitContext* ctx);

// Core operations
int onebit_forward(OneBitContext* ctx, 
                  const float* input,
                  float* output,
                  int batch_size,
                  int seq_len);

int onebit_backward(OneBitContext* ctx,
                   const float* grad_output,
                   float* grad_input,
                   int batch_size,
                   int seq_len);

// State management
int onebit_save_state(OneBitContext* ctx, const char* path);
int onebit_load_state(OneBitContext* ctx, const char* path);
int onebit_reset_state(OneBitContext* ctx);

// Performance monitoring
typedef struct {
    double forward_time;
    double backward_time;
    double memory_used;
    double throughput;
    int active_batches;
    int cached_sequences;
} OneBitStats;

int onebit_get_stats(OneBitContext* ctx, OneBitStats* stats);

#endif 