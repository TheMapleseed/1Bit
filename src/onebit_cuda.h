#ifndef ONEBIT_CUDA_H
#define ONEBIT_CUDA_H

#include "onebit_quant.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA context
typedef struct {
    cublasHandle_t cublas_handle;
    cudaStream_t stream;
    int device_id;
    size_t available_memory;
    void* workspace;
    size_t workspace_size;
} OneBitCUDAContext;

// CUDA tensor wrapper
typedef struct {
    void* data;
    uint32_t dims[4];
    uint32_t type;
    size_t size;
    bool is_device_memory;
} CUDATensor;

// Function declarations
int init_cuda_context(OneBitCUDAContext* ctx, int device_id);
void destroy_cuda_context(OneBitCUDAContext* ctx);

// Memory management
int cuda_memcpy_to_device(void* dst, const void* src, size_t size);
int cuda_memcpy_to_host(void* dst, const void* src, size_t size);

// Tensor operations
int cuda_tensor_create(CUDATensor* tensor, const uint32_t* dims,
                      uint32_t type, bool allocate);
int cuda_tensor_from_host(CUDATensor* tensor, const QuantizedTensor* host_tensor);
int cuda_tensor_to_host(const CUDATensor* tensor, QuantizedTensor* host_tensor);
void cuda_tensor_destroy(CUDATensor* tensor);

// CUDA kernels
int cuda_matrix_multiply(OneBitCUDAContext* ctx,
                        const CUDATensor* A,
                        const CUDATensor* B,
                        CUDATensor* C,
                        bool transpose_a,
                        bool transpose_b);
int cuda_layer_norm(OneBitCUDAContext* ctx,
                   CUDATensor* input,
                   const CUDATensor* weight,
                   const CUDATensor* bias,
                   float eps);
int cuda_softmax(OneBitCUDAContext* ctx,
                 CUDATensor* input,
                 int dim);

// Error handling
const char* cuda_get_error_string(int error);
#define CUDA_CHECK(x) 