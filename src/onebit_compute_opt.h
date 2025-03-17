/**
 * @file onebit_compute_opt.h
 * @brief High-performance compute optimizations for OneBit
 *
 * Provides SIMD-optimized operations, GPU acceleration, and parallel processing
 * capabilities for neural network computations. Includes automatic hardware
 * detection and optimal dispatch.
 *
 * @author OneBit Team
 * @version 1.0
 */

#ifndef ONEBIT_COMPUTE_OPT_H
#define ONEBIT_COMPUTE_OPT_H

#include <stdint.h>
#include <stdbool.h>

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
} HardwareCapabilities;

// Compute context
typedef struct {
    HardwareCapabilities hw_caps;
    void* gpu_contexts[8];
    void* cuda_streams[8];
    void* compute_buffers;
    size_t buffer_size;
    int current_device;
    bool mixed_precision;
    void* workspace;
    size_t workspace_size;
} ComputeContext;

// Matrix operation descriptors
typedef struct {
    void* data;
    size_t rows;
    size_t cols;
    size_t stride;
    bool transpose;
    bool on_device;
    int device_id;
} MatrixDescriptor;

// Initialize and cleanup
int compute_init(ComputeContext* ctx);
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

// Memory operations
int compute_memcpy_h2d(ComputeContext* ctx,
                      void* dst,
                      const void* src,
                      size_t size);

int compute_memcpy_d2h(ComputeContext* ctx,
                      void* dst,
                      const void* src,
                      size_t size);

// Utility functions
const char* compute_get_error_string(int error_code);
void compute_print_capabilities(const ComputeContext* ctx);

#endif /* ONEBIT_COMPUTE_OPT_H */ 