/**
 * @file onebit_compute.h
 * @brief Hardware-optimized compute operations for OneBit
 *
 * Provides hardware-specific optimizations for neural network operations,
 * including SIMD (AVX-512, AVX2, SSE4) and GPU acceleration (CUDA, ROCm).
 *
 * @author OneBit Team
 * @version 1.0.0
 */

#ifndef ONEBIT_onebit_compute_H
#define ONEBIT_onebit_compute_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Hardware capabilities
typedef struct {
    bool has_avx512;
    bool has_avx2;
    bool has_sse4;
    bool has_neon;
    int num_cpu_cores;
    int num_gpu_devices;
    size_t gpu_memory[8];  // Support up to 8 GPUs
    bool cuda_available;
    bool rocm_available;
    bool metal_available;
} HardwareCapabilities;

// Matrix descriptor
typedef struct {
    void* data;
    size_t rows;
    size_t cols;
    size_t stride;
    bool transpose;
    bool on_device;
    int device_id;
} MatrixDescriptor;

// Compute context (opaque)
struct ComputeContextStruct {
    HardwareCapabilities hw_caps;
    void* gpu_contexts[8];
    void* streams[8];
    void* compute_buffers;
    size_t buffer_size;
    int current_device;
    bool mixed_precision;
    void* workspace;
    size_t workspace_size;
};

// Initialization and cleanup
int compute_init(ComputeContext* ctx, bool use_cuda, bool use_rocm, bool use_metal);
void compute_cleanup(ComputeContext* ctx);

// Device management
int compute_set_device(ComputeContext* ctx, int device_id);
int compute_sync_device(ComputeContext* ctx, int device_id);
int compute_memory_info(ComputeContext* ctx, size_t* free, size_t* total);

// Core compute operations
int compute_gemm(ComputeContext* ctx,
                const MatrixDescriptor* A,
                const MatrixDescriptor* B,
                MatrixDescriptor* C,
                float alpha,
                float beta);

int compute_binary_gemm(ComputeContext* ctx,
                       const MatrixDescriptor* A,
                       const MatrixDescriptor* B,
                       MatrixDescriptor* C,
                       float scale);

int compute_quantize(ComputeContext* ctx,
                    const MatrixDescriptor* input,
                    MatrixDescriptor* output,
                    float scale);

int compute_dequantize(ComputeContext* ctx,
                      const MatrixDescriptor* input,
                      MatrixDescriptor* output,
                      float scale);

// Layer operations
int compute_layer_norm(ComputeContext* ctx,
                      const MatrixDescriptor* input,
                      const MatrixDescriptor* gamma,
                      const MatrixDescriptor* beta,
                      MatrixDescriptor* output,
                      float epsilon);

int compute_softmax(ComputeContext* ctx,
                   const MatrixDescriptor* input,
                   MatrixDescriptor* output,
                   int dim);

int compute_attention(ComputeContext* ctx,
                     const MatrixDescriptor* query,
                     const MatrixDescriptor* key,
                     const MatrixDescriptor* value,
                     MatrixDescriptor* output,
                     float scale,
                     const MatrixDescriptor* mask);

// Memory operations
int compute_memcpy_h2d(ComputeContext* ctx,
                      void* dst,
                      const void* src,
                      size_t size);

int compute_memcpy_d2h(ComputeContext* ctx,
                      void* dst,
                      const void* src,
                      size_t size);

int compute_memcpy_d2d(ComputeContext* ctx,
                      void* dst,
                      const void* src,
                      size_t size);

// Utility functions
const char* compute_get_error_string(int error_code);
void compute_print_capabilities(const ComputeContext* ctx);
int compute_get_device_count(void);
int compute_get_device_info(int device_id, char* name, size_t* memory);

#endif /* ONEBIT_COMPUTE_H */ 