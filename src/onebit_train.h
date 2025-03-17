/**
 * @file onebit_train.h
 * @brief High-performance training system
 */

#ifndef ONEBIT_TRAIN_H
#define ONEBIT_TRAIN_H

#include "onebit_model.h"
#include "onebit_data.h"
#include <stdint.h>

// Training configuration
typedef struct {
    // Optimization parameters
    float learning_rate;
    float weight_decay;
    float grad_clip;
    int batch_size;
    int max_epochs;
    
    // Mixed precision training
    bool use_amp;
    float loss_scale;
    float loss_scale_window;
    
    // Checkpointing
    int checkpoint_interval;
    char* checkpoint_dir;
    bool save_optimizer_state;
    
    // Monitoring
    bool enable_profiling;
    bool log_gradients;
    int print_frequency;
    
    // Hardware utilization
    int num_workers;
    bool pin_memory;
    size_t prefetch_factor;
} TrainConfig;

// Training state
typedef struct {
    ModelContext* model;
    void* optimizer;
    void* scheduler;
    void* scaler;
    void* profiler;
    DataIterator* train_data;
    DataIterator* valid_data;
    uint64_t global_step;
    float best_loss;
    float last_loss;
    float best_metric;
    uint64_t epoch;
    bool early_stopped;
    char* checkpoint_path;
} TrainState;

// Online learning configuration
typedef struct {
    bool enabled;                // Whether online learning is enabled
    float learning_rate;         // Learning rate for online updates (smaller than training)
    int update_frequency;        // How often to apply updates (e.g., every N inferences)
    float forgetting_factor;     // Weight for old vs. new knowledge (0-1)
    bool cache_examples;         // Whether to cache examples for batch updates
    int cache_size;              // Maximum number of examples to cache
    bool adaptive_lr;            // Whether to use adaptive learning rate
    LossType loss_type;          // Loss function to use
    bool update_during_inference;// Whether to update during inference or afterwards
} OnlineLearningConfig;

// API functions

// Initialize training
int onebit_train_init(OneBitContext* ctx, TrainConfig* config, TrainState** state);

// Run full training loop
int onebit_train(OneBitContext* ctx, TrainState* state, Dataset* train_data, Dataset* val_data);

// Train for a single epoch
int onebit_train_epoch(OneBitContext* ctx, TrainState* state);

// Run validation
float onebit_validate(OneBitContext* ctx, TrainState* state);

// Save checkpoint
int onebit_save_checkpoint(OneBitContext* ctx, TrainState* state, const char* path);

// Load checkpoint
int onebit_load_checkpoint(OneBitContext* ctx, TrainState* state, const char* path);

// Clean up training state
void onebit_train_cleanup(TrainState* state);

// Online learning functions
int onebit_enable_online_learning(OneBitContext* ctx, OnlineLearningConfig* config);

// Update model from feedback during or after inference
int onebit_online_update(OneBitContext* ctx, const float* input, const float* target, float* loss);

// Reset online learning state (e.g., cached examples)
void onebit_reset_online_learning(OneBitContext* ctx);

// Get online learning statistics
int onebit_get_online_learning_stats(OneBitContext* ctx, float* loss, float* learning_rate, int* updates);

#endif // ONEBIT_TRAIN_H 