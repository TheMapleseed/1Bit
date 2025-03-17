#ifndef ONEBIT_MODEL_H
#define ONEBIT_MODEL_H

#include "onebit_config.h"
#include "onebit_compute_opt.h"
#include <stdint.h>

// Model parameter types
typedef enum {
    PARAM_TYPE_FLOAT32,
    PARAM_TYPE_FLOAT16,
    PARAM_TYPE_INT8,
    PARAM_TYPE_INT4,
    PARAM_TYPE_BINARY
} ParamType;

// Parameter tensor
typedef struct {
    void* data;
    size_t* shape;
    size_t ndim;
    ParamType type;
    bool requires_grad;
    void* grad;
} Parameter;

// Model context
typedef struct {
    Parameter* parameters;
    size_t num_parameters;
    void* forward_state;
    void* backward_state;
    void* optimizer_state;
    OneBitConfig config;
    ComputeContext* compute;
} ModelContext;

// Function declarations
int model_init(ModelContext* ctx, const OneBitConfig* config);
void model_cleanup(ModelContext* ctx);

// Model operations
int model_load_weights(ModelContext* ctx, const char* path);
int model_save_weights(ModelContext* ctx, const char* path);
int model_to_device(ModelContext* ctx, int device_id);

// Training operations
int model_forward(ModelContext* ctx, void* input, void* output);
int model_backward(ModelContext* ctx, void* grad_output);
int model_update(ModelContext* ctx);
int model_zero_grad(ModelContext* ctx);

// State management
int model_save_checkpoint(ModelContext* ctx, const char* path);
int model_load_checkpoint(ModelContext* ctx, const char* path);
int model_export(ModelContext* ctx, const char* path, const char* format);

// Utility functions
size_t model_get_num_parameters(const ModelContext* ctx);
void model_print_summary(const ModelContext* ctx);
int model_verify_weights(const ModelContext* ctx);

#endif 