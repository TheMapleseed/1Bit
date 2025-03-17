#ifndef ONEBIT_CONFIG_H
#define ONEBIT_CONFIG_H

#include <stdint.h>
#include <stdbool.h>

// Model architecture configuration
typedef struct {
    int num_layers;
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int max_position_embeddings;
    int vocab_size;
    float attention_dropout;
    float hidden_dropout;
    bool use_bias;
    char* activation_function;
} ModelConfig;

// Training configuration
typedef struct {
    float learning_rate;
    float weight_decay;
    int warmup_steps;
    int max_steps;
    int batch_size;
    int gradient_accumulation_steps;
    bool use_mixed_precision;
    char* optimizer_type;
} TrainingConfig;

// Runtime configuration
typedef struct {
    bool use_cuda;
    int device_id;
    int num_threads;
    size_t memory_limit;
    bool enable_profiling;
    char* log_level;
} RuntimeConfig;

// Configuration context
typedef struct {
    ModelConfig model;
    TrainingConfig training;
    RuntimeConfig runtime;
    char* model_path;
    char* data_path;
    char* output_path;
} ConfigContext;

// Function declarations
int config_init(ConfigContext* ctx);
void config_cleanup(ConfigContext* ctx);

// File operations
int config_load_json(ConfigContext* ctx, const char* path);
int config_save_json(const ConfigContext* ctx, const char* path);

// Validation and utilities
bool config_validate(const ConfigContext* ctx);
void config_print(const ConfigContext* ctx);
char* config_to_string(const ConfigContext* ctx);

// Default configurations
int config_set_defaults(ConfigContext* ctx);
int config_merge(ConfigContext* dest, const ConfigContext* src);

#endif 