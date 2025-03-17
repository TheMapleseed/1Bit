#ifndef ONEBIT_QUANT_H
#define ONEBIT_QUANT_H

#include <stdint.h>

// Tensor types
#define TENSOR_TYPE_FLOAT32 0
#define TENSOR_TYPE_INT8    1
#define TENSOR_TYPE_UINT8   2

// Quantization parameters
typedef struct {
    float scale;
    float zero_point;
    float min_val;
    float max_val;
} QuantParams;

// Quantized tensor
typedef struct {
    void* data;
    float* scales;
    uint32_t dims[4];
    size_t size;
    size_t num_scales;
    uint32_t type;
} QuantizedTensor;

// Function declarations
int quantize_tensor(const float* input, QuantizedTensor* output,
                   const uint32_t* dims, int num_dims);
int dequantize_tensor(const QuantizedTensor* input, float* output);
int convert_precision(QuantizedTensor* tensor, uint32_t new_type);

// Helper functions
void compute_quant_params(const float* data, size_t size,
                         QuantParams* params);
void apply_quantization(const float* input, void* output,
                       size_t size, const QuantParams* params,
                       uint32_t type);
void apply_dequantization(const void* input, float* output,
                         size_t size, const QuantParams* params,
                         uint32_t type);
size_t get_type_size(uint32_t type);

#endif 