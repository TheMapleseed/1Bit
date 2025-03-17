/**
 * @file onebit_model.h
 * @brief Neural network model management for OneBit
 *
 * Provides model loading, inference, and parameter management functionality.
 * Supports different parameter types and hardware acceleration.
 *
 * @author OneBit Team
 * @version 1.0.0
 */

#ifndef ONEBIT_onebit_model_H
#define ONEBIT_onebit_model_H

#include "onebit_compute.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Forward declarations
typedef struct OneBitConfigStruct OneBitConfig;
typedef struct ComputeContextStruct ComputeContext;

// Parameter tensor
typedef struct {
    void* data;
    size_t* shape;
    size_t ndim;
    ParamType type;
    bool requires_grad;
    void* grad;
} Parameter;

// Model structure (opaque)
struct OneBitModelStruct {
    Parameter* parameters;
    size_t num_parameters;
    void* forward_state;
    void* backward_state;
    void* optimizer_state;
    OneBitConfig config;
    ComputeContext* compute;
};

// Function declarations
int model_init(OneBitModel* model, const OneBitConfig* config);
void model_cleanup(OneBitModel* model);

// Model operations
int model_load_weights(OneBitModel* model, const char* path);
int model_save_weights(OneBitModel* model, const char* path);
int model_to_device(OneBitModel* model, int device_id);

// Inference operations
int model_forward(OneBitModel* model, const float* input, float* output, size_t batch_size, size_t seq_len);
int model_generate(OneBitModel* model, const int* input_tokens, size_t input_length, 
                  int* output_tokens, size_t max_output_length,
                  float temperature, float top_p);

// Training operations
int model_backward(OneBitModel* model, const float* grad_output);
int model_update(OneBitModel* model);
int model_zero_grad(OneBitModel* model);

// State management
int model_save_checkpoint(OneBitModel* model, const char* path);
int model_load_checkpoint(OneBitModel* model, const char* path);
int model_export(OneBitModel* model, const char* path, const char* format);

// Quantization
int model_quantize(OneBitModel* model, ParamType target_type);
int model_dequantize(OneBitModel* model);

// Utility functions
size_t model_get_num_parameters(const OneBitModel* model);
void model_print_summary(const OneBitModel* model);
int model_verify_weights(const OneBitModel* model);

#endif /* ONEBIT_MODEL_H */ 