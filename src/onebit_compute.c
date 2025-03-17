#include "onebit/onebit_compute.h"
#include "onebit/onebit_compute_cuda.h"
#include "onebit/onebit_compute_metal.h"
#include <stdlib.h>
#include <string.h>

// Global compute state
static struct {
    bool initialized;
    ComputeDeviceType active_device;
    void* device_context;  // Points to CUDA or Metal context
    char last_error[256];
} g_compute_state = {0};

// Error handling
static void set_error(const char* msg) {
    strncpy(g_compute_state.last_error, msg, sizeof(g_compute_state.last_error) - 1);
}

// Device detection and selection
static ComputeDeviceType detect_best_device(void) {
    #ifdef HAVE_CUDA
    // Check for CUDA device
    int cuda_devices = 0;
    if (cudaGetDeviceCount(&cuda_devices) == cudaSuccess && cuda_devices > 0) {
        return COMPUTE_DEVICE_CUDA;
    }
    #endif

    #ifdef __APPLE__
    // Check for Metal device
    if (@available(macOS 10.13, *)) {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device) {
            [device release];
            return COMPUTE_DEVICE_METAL;
        }
    }
    #endif

    // Fallback to CPU
    return COMPUTE_DEVICE_CPU;
}

int onebit_init(void) {
    // Prevent double initialization
    if (g_compute_state.initialized) {
        set_error("Already initialized");
        return -1;
    }

    // Detect best available device
    g_compute_state.active_device = detect_best_device();

    // Initialize device-specific context
    switch (g_compute_state.active_device) {
        case COMPUTE_DEVICE_CUDA: {
            #ifdef HAVE_CUDA
            cudaError_t err = cuda_init_context();
            if (err != cudaSuccess) {
                set_error("CUDA initialization failed");
                return -1;
            }
            #else
            set_error("CUDA support not compiled in");
            return -1;
            #endif
            break;
        }

        case COMPUTE_DEVICE_METAL: {
            #ifdef __APPLE__
            int err = metal_init_context();
            if (err != 0) {
                set_error("Metal initialization failed");
                return -1;
            }
            #else
            set_error("Metal support not available");
            return -1;
            #endif
            break;
        }

        case COMPUTE_DEVICE_CPU:
            // CPU initialization (thread pool, SIMD detection, etc.)
            g_compute_state.device_context = NULL;  // CPU doesn't need special context
            break;

        default:
            set_error("Unknown compute device type");
            return -1;
    }

    g_compute_state.initialized = true;
    return 0;
}

void onebit_cleanup(void) {
    if (!g_compute_state.initialized) {
        return;
    }

    // Cleanup device-specific resources
    switch (g_compute_state.active_device) {
        case COMPUTE_DEVICE_CUDA:
            #ifdef HAVE_CUDA
            cudaDeviceReset();
            #endif
            break;

        case COMPUTE_DEVICE_METAL:
            #ifdef __APPLE__
            metal_cleanup_context();
            #endif
            break;

        case COMPUTE_DEVICE_CPU:
            // CPU cleanup (thread pool, SIMD detection, etc.)
            break;

        default:
            break;
    }

    g_compute_state.initialized = false;
}

// CPU detection functions
static void detect_cpu_capabilities(HardwareCapabilities* caps) {
    // This is a placeholder implementation
    // In a real implementation, this would use CPUID or similar mechanisms
    
#ifdef ONEBIT_ENABLE_AVX512
    caps->has_avx512 = true;
#else
    caps->has_avx512 = false;
#endif

#ifdef ONEBIT_ENABLE_AVX2
    caps->has_avx2 = true;
#else
    caps->has_avx2 = false;
#endif

#ifdef ONEBIT_ENABLE_SSE4
    caps->has_sse4 = true;
#else
    caps->has_sse4 = false;
#endif

    // ARM NEON detection would go here
    caps->has_neon = false;
    
    // Get number of CPU cores
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    caps->num_cpu_cores = sysinfo.dwNumberOfProcessors;
#else
    // For Unix-like systems
    caps->num_cpu_cores = sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

// CUDA device detection
static void detect_cuda_devices(HardwareCapabilities* caps) {
#ifdef ONEBIT_USE_CUDA
    // In a real implementation, this would enumerate CUDA devices
    // For now, just set placeholder values
    caps->cuda_available = true;
    caps->num_gpu_devices = 1;
    caps->gpu_memory[0] = 8ULL * 1024 * 1024 * 1024;  // 8GB
#else
    caps->cuda_available = false;
    caps->num_gpu_devices = 0;
#endif
}

// ROCm device detection
static void detect_rocm_devices(HardwareCapabilities* caps) {
#ifdef ONEBIT_USE_ROCM
    // In a real implementation, this would enumerate ROCm devices
    // For now, just set placeholder values
    caps->rocm_available = true;
    caps->num_gpu_devices = 1;
    caps->gpu_memory[0] = 16ULL * 1024 * 1024 * 1024;  // 16GB
#else
    caps->rocm_available = false;
#endif
}

// Metal device detection
static void detect_metal_devices(HardwareCapabilities* caps) {
#if defined(ONEBIT_USE_METAL) && defined(__APPLE__)
    // In a real implementation, this would enumerate Metal devices
    // For now, just set placeholder values
    caps->metal_available = true;
    caps->num_gpu_devices = 1;
    caps->gpu_memory[0] = 8ULL * 1024 * 1024 * 1024;  // 8GB
#else
    caps->metal_available = false;
#endif
}

// Initialize compute context
int compute_init(ComputeContext* ctx, bool use_cuda, bool use_rocm, bool use_metal) {
    if (!ctx) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Clear the context
    memset(ctx, 0, sizeof(ComputeContext));
    
    // Detect hardware capabilities
    detect_cpu_capabilities(&ctx->hw_caps);
    
    // Detect GPU devices based on requested backends
    if (use_cuda) {
        detect_cuda_devices(&ctx->hw_caps);
    }
    
    if (use_rocm) {
        detect_rocm_devices(&ctx->hw_caps);
    }
    
    if (use_metal) {
        detect_metal_devices(&ctx->hw_caps);
    }
    
    // Set default device
    ctx->current_device = -1;  // CPU
    
    // Allocate workspace
    ctx->workspace_size = 64 * 1024 * 1024;  // 64MB workspace
    ctx->workspace = malloc(ctx->workspace_size);
    if (!ctx->workspace) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Initialize GPU contexts if available
    if (ctx->hw_caps.cuda_available && use_cuda) {
        // CUDA initialization would go here
        // For now, just set placeholder values
        ctx->gpu_contexts[0] = malloc(1);  // Dummy allocation
        if (!ctx->gpu_contexts[0]) {
            free(ctx->workspace);
            return ONEBIT_ERROR_MEMORY;
        }
        
        ctx->streams[0] = malloc(1);  // Dummy allocation
        if (!ctx->streams[0]) {
            free(ctx->gpu_contexts[0]);
            free(ctx->workspace);
            return ONEBIT_ERROR_MEMORY;
        }
        
        ctx->current_device = 0;
    }
    else if (ctx->hw_caps.rocm_available && use_rocm) {
        // ROCm initialization would go here
        // Similar to CUDA initialization
        ctx->current_device = 0;
    }
    else if (ctx->hw_caps.metal_available && use_metal) {
        // Metal initialization would go here
        // Similar to CUDA initialization
        ctx->current_device = 0;
    }
    
    return ONEBIT_SUCCESS;
}

// Clean up compute context
void compute_cleanup(ComputeContext* ctx) {
    if (!ctx) return;
    
    // Free GPU contexts
    for (int i = 0; i < 8; i++) {
        if (ctx->gpu_contexts[i]) {
            free(ctx->gpu_contexts[i]);
            ctx->gpu_contexts[i] = NULL;
        }
        
        if (ctx->streams[i]) {
            free(ctx->streams[i]);
            ctx->streams[i] = NULL;
        }
    }
    
    // Free workspace
    free(ctx->workspace);
    ctx->workspace = NULL;
}

// Set current device
int compute_set_device(ComputeContext* ctx, int device_id) {
    if (!ctx) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    if (device_id >= 0 && device_id >= ctx->hw_caps.num_gpu_devices) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->current_device = device_id;
    
    return ONEBIT_SUCCESS;
}

// Synchronize device
int compute_sync_device(ComputeContext* ctx, int device_id) {
    if (!ctx) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    if (device_id >= 0 && device_id >= ctx->hw_caps.num_gpu_devices) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // In a real implementation, this would synchronize the device
    // For now, just return success
    
    return ONEBIT_SUCCESS;
}

// Get device memory information
int compute_memory_info(ComputeContext* ctx, size_t* free, size_t* total) {
    if (!ctx || !free || !total) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    if (ctx->current_device < 0) {
        // CPU memory
        *free = 1024 * 1024 * 1024;  // 1GB (placeholder)
        *total = 16 * 1024 * 1024 * 1024;  // 16GB (placeholder)
    } else if (ctx->current_device < ctx->hw_caps.num_gpu_devices) {
        // GPU memory
        *free = ctx->hw_caps.gpu_memory[ctx->current_device] / 2;  // Half free (placeholder)
        *total = ctx->hw_caps.gpu_memory[ctx->current_device];
    } else {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    return ONEBIT_SUCCESS;
}

// Matrix multiplication (GEMM)
int compute_gemm(ComputeContext* ctx,
                const MatrixDescriptor* A,
                const MatrixDescriptor* B,
                MatrixDescriptor* C,
                float alpha,
                float beta) {
    if (!ctx || !A || !B || !C) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Check dimensions
    if (A->cols != B->rows) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    if (C->rows != A->rows || C->cols != B->cols) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Select implementation based on hardware and device
    if (ctx->current_device >= 0) {
        // GPU implementation
        // In a real implementation, this would dispatch to CUDA/ROCm/Metal
        // For now, just return success
    } else {
        // CPU implementation
        // In a real implementation, this would dispatch to SIMD optimized code
        // For now, just do a simple implementation
        
        // Get raw pointers (assumes row-major ordering)
        const float* a_data = (const float*)A->data;
        const float* b_data = (const float*)B->data;
        float* c_data = (float*)C->data;
        
        // Simple triple loop matrix multiplication
        for (size_t i = 0; i < A->rows; i++) {
            for (size_t j = 0; j < B->cols; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < A->cols; k++) {
                    sum += a_data[i * A->stride + k] * b_data[k * B->stride + j];
                }
                c_data[i * C->stride + j] = alpha * sum + beta * c_data[i * C->stride + j];
            }
        }
    }
    
    return ONEBIT_SUCCESS;
}

// Binary matrix multiplication (binary GEMM)
int compute_binary_gemm(ComputeContext* ctx,
                       const MatrixDescriptor* A,
                       const MatrixDescriptor* B,
                       MatrixDescriptor* C,
                       float scale) {
    if (!ctx || !A || !B || !C) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Check dimensions
    if (A->cols != B->rows) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    if (C->rows != A->rows || C->cols != B->cols) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Select implementation based on hardware and device
    if (ctx->current_device >= 0) {
        // GPU implementation
        // In a real implementation, this would dispatch to CUDA/ROCm/Metal
        // For now, just return success
    } else {
        // CPU implementation
        // In a real implementation, this would use bit operations for efficiency
        // For now, just simulate binary operations
        
        // Get raw pointers (assumes row-major ordering)
        const uint8_t* a_data = (const uint8_t*)A->data;
        const uint8_t* b_data = (const uint8_t*)B->data;
        float* c_data = (float*)C->data;
        
        // Simple triple loop matrix multiplication with binary operations
        for (size_t i = 0; i < A->rows; i++) {
            for (size_t j = 0; j < B->cols; j++) {
                int sum = 0;
                for (size_t k = 0; k < A->cols; k++) {
                    // Simulate binary operation (actual implementation would use bitwise ops)
                    uint8_t a_bit = (a_data[i * A->stride + k / 8] >> (k % 8)) & 1;
                    uint8_t b_bit = (b_data[k * B->stride + j / 8] >> (j % 8)) & 1;
                    sum += (a_bit == b_bit) ? 1 : -1;  // XNOR operation
                }
                c_data[i * C->stride + j] = sum * scale;
            }
        }
    }
    
    return ONEBIT_SUCCESS;
}

// Quantize floating point values
int compute_quantize(ComputeContext* ctx,
                    const MatrixDescriptor* input,
                    MatrixDescriptor* output,
                    float scale) {
    if (!ctx || !input || !output) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Check dimensions
    if (input->rows != output->rows || input->cols != output->cols) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Select implementation based on hardware and device
    if (ctx->current_device >= 0) {
        // GPU implementation
        // In a real implementation, this would dispatch to CUDA/ROCm/Metal
        // For now, just return success
    } else {
        // CPU implementation
        // Get raw pointers
        const float* input_data = (const float*)input->data;
        uint8_t* output_data = (uint8_t*)output->data;
        
        // Simple quantization (for binary values)
        size_t total_elements = input->rows * input->cols;
        
        // Binary quantization (1-bit)
        for (size_t i = 0; i < total_elements; i += 8) {
            uint8_t byte = 0;
            
            // Pack 8 binary values into a byte
            for (int j = 0; j < 8 && (i + j) < total_elements; j++) {
                float val = input_data[i + j];
                if (val > 0) {
                    byte |= (1 << j);
                }
            }
            
            output_data[i / 8] = byte;
        }
    }
    
    return ONEBIT_SUCCESS;
}

// Dequantize values back to floating point
int compute_dequantize(ComputeContext* ctx,
                      const MatrixDescriptor* input,
                      MatrixDescriptor* output,
                      float scale) {
    if (!ctx || !input || !output) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Select implementation based on hardware and device
    if (ctx->current_device >= 0) {
        // GPU implementation
        // In a real implementation, this would dispatch to CUDA/ROCm/Metal
        // For now, just return success
    } else {
        // CPU implementation
        // Get raw pointers
        const uint8_t* input_data = (const uint8_t*)input->data;
        float* output_data = (float*)output->data;
        
        // Simple dequantization (for binary values)
        size_t total_elements = output->rows * output->cols;
        
        // Binary dequantization (1-bit)
        for (size_t i = 0; i < total_elements; i++) {
            uint8_t byte = input_data[i / 8];
            int bit = (byte >> (i % 8)) & 1;
            output_data[i] = bit ? scale : -scale;
        }
    }
    
    return ONEBIT_SUCCESS;
}

// Layer normalization
int compute_layer_norm(ComputeContext* ctx,
                      const MatrixDescriptor* input,
                      const MatrixDescriptor* gamma,
                      const MatrixDescriptor* beta,
                      MatrixDescriptor* output,
                      float epsilon) {
    if (!ctx || !input || !gamma || !beta || !output) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Check dimensions
    if (input->rows != output->rows || input->cols != output->cols) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    if (gamma->cols != input->cols || beta->cols != input->cols) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Select implementation based on hardware and device
    if (ctx->current_device >= 0) {
        // GPU implementation
        // In a real implementation, this would dispatch to CUDA/ROCm/Metal
        // For now, just return success
    } else {
        // CPU implementation
        // Get raw pointers
        const float* input_data = (const float*)input->data;
        const float* gamma_data = (const float*)gamma->data;
        const float* beta_data = (const float*)beta->data;
        float* output_data = (float*)output->data;
        
        // Layer normalization along last dimension
        for (size_t i = 0; i < input->rows; i++) {
            // Compute mean
            float mean = 0.0f;
            for (size_t j = 0; j < input->cols; j++) {
                mean += input_data[i * input->stride + j];
            }
            mean /= input->cols;
            
            // Compute variance
            float var = 0.0f;
            for (size_t j = 0; j < input->cols; j++) {
                float diff = input_data[i * input->stride + j] - mean;
                var += diff * diff;
            }
            var /= input->cols;
            
            // Normalize and scale
            for (size_t j = 0; j < input->cols; j++) {
                float normalized = (input_data[i * input->stride + j] - mean) / sqrtf(var + epsilon);
                output_data[i * output->stride + j] = normalized * gamma_data[j] + beta_data[j];
            }
        }
    }
    
    return ONEBIT_SUCCESS;
}

// Softmax operation
int compute_softmax(ComputeContext* ctx,
                   const MatrixDescriptor* input,
                   MatrixDescriptor* output,
                   int dim) {
    if (!ctx || !input || !output) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Check dimensions
    if (input->rows != output->rows || input->cols != output->cols) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Select implementation based on hardware and device
    if (ctx->current_device >= 0) {
        // GPU implementation
        // In a real implementation, this would dispatch to CUDA/ROCm/Metal
        // For now, just return success
    } else {
        // CPU implementation
        // Get raw pointers
        const float* input_data = (const float*)input->data;
        float* output_data = (float*)output->data;
        
        // Only support dim = 1 (along columns) for simplicity
        if (dim != 1) {
            return ONEBIT_ERROR_NOT_SUPPORTED;
        }
        
        // Apply softmax per row
        for (size_t i = 0; i < input->rows; i++) {
            // Find max value for numerical stability
            float max_val = input_data[i * input->stride];
            for (size_t j = 1; j < input->cols; j++) {
                float val = input_data[i * input->stride + j];
                if (val > max_val) {
                    max_val = val;
                }
            }
            
            // Compute exponentials and sum
            float sum = 0.0f;
            for (size_t j = 0; j < input->cols; j++) {
                output_data[i * output->stride + j] = expf(input_data[i * input->stride + j] - max_val);
                sum += output_data[i * output->stride + j];
            }
            
            // Normalize
            for (size_t j = 0; j < input->cols; j++) {
                output_data[i * output->stride + j] /= sum;
            }
        }
    }
    
    return ONEBIT_SUCCESS;
}

// Attention computation
int compute_attention(ComputeContext* ctx,
                     const MatrixDescriptor* query,
                     const MatrixDescriptor* key,
                     const MatrixDescriptor* value,
                     MatrixDescriptor* output,
                     float scale,
                     const MatrixDescriptor* mask) {
    if (!ctx || !query || !key || !value || !output) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Check dimensions (simplified)
    if (query->cols != key->cols) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Select implementation based on hardware and device
    if (ctx->current_device >= 0) {
        // GPU implementation
        // In a real implementation, this would dispatch to CUDA/ROCm/Metal
        // For now, just return success
    } else {
        // CPU implementation
        // This is a very simplified attention implementation
        // A real implementation would handle batching, multiple heads, etc.
        
        // Get raw pointers
        const float* q_data = (const float*)query->data;
        const float* k_data = (const float*)key->data;
        const float* v_data = (const float*)value->data;
        float* out_data = (float*)output->data;
        
        // Compute attention scores (Q*K^T)
        float* scores = (float*)malloc(query->rows * key->rows * sizeof(float));
        if (!scores) {
            return ONEBIT_ERROR_MEMORY;
        }
        
        for (size_t i = 0; i < query->rows; i++) {
            for (size_t j = 0; j < key->rows; j++) {
                float dot = 0.0f;
                for (size_t k = 0; k < query->cols; k++) {
                    dot += q_data[i * query->stride + k] * k_data[j * key->stride + k];
                }
                scores[i * key->rows + j] = dot * scale;
            }
        }
        
        // Apply mask if provided
        if (mask) {
            const float* mask_data = (const float*)mask->data;
            for (size_t i = 0; i < query->rows; i++) {
                for (size_t j = 0; j < key->rows; j++) {
                    scores[i * key->rows + j] += mask_data[i * mask->stride + j];
                }
            }
        }
        
        // Apply softmax to get attention weights
        for (size_t i = 0; i < query->rows; i++) {
            // Find max for numerical stability
            float max_val = scores[i * key->rows];
            for (size_t j = 1; j < key->rows; j++) {
                if (scores[i * key->rows + j] > max_val) {
                    max_val = scores[i * key->rows + j];
                }
            }
            
            // Compute exp and sum
            float sum = 0.0f;
            for (size_t j = 0; j < key->rows; j++) {
                scores[i * key->rows + j] = expf(scores[i * key->rows + j] - max_val);
                sum += scores[i * key->rows + j];
            }
            
            // Normalize
            for (size_t j = 0; j < key->rows; j++) {
                scores[i * key->rows + j] /= sum;
            }
        }
        
        // Compute output (attention_weights * V)
        for (size_t i = 0; i < query->rows; i++) {
            for (size_t j = 0; j < value->cols; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < key->rows; k++) {
                    sum += scores[i * key->rows + k] * v_data[k * value->stride + j];
                }
                out_data[i * output->stride + j] = sum;
            }
        }
        
        free(scores);
    }
    
    return ONEBIT_SUCCESS;
}

// Memory operations: host to device
int compute_memcpy_h2d(ComputeContext* ctx, void* dst, const void* src, size_t size) {
    if (!ctx || !dst || !src || size == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    if (ctx->current_device < 0) {
        // CPU to CPU copy
        memcpy(dst, src, size);
    } else {
        // CPU to GPU copy
        // In a real implementation, this would use cudaMemcpy or equivalent
        // For now, just simulate the copy
        memcpy(dst, src, size);
    }
    
    return ONEBIT_SUCCESS;
}

// Memory operations: device to host
int compute_memcpy_d2h(ComputeContext* ctx, void* dst, const void* src, size_t size) {
    if (!ctx || !dst || !src || size == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    if (ctx->current_device < 0) {
        // CPU to CPU copy
        memcpy(dst, src, size);
    } else {
        // GPU to CPU copy
        // In a real implementation, this would use cudaMemcpy or equivalent
        // For now, just simulate the copy
        memcpy(dst, src, size);
    }
    
    return ONEBIT_SUCCESS;
}

// Memory operations: device to device
int compute_memcpy_d2d(ComputeContext* ctx, void* dst, const void* src, size_t size) {
    if (!ctx || !dst || !src || size == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    if (ctx->current_device < 0) {
        // CPU to CPU copy
        memcpy(dst, src, size);
    } else {
        // GPU to GPU copy
        // In a real implementation, this would use cudaMemcpy or equivalent
        // For now, just simulate the copy
        memcpy(dst, src, size);
    }
    
    return ONEBIT_SUCCESS;
}

// Get error string
const char* compute_get_error_string(int error_code) {
    // This should map compute-specific error codes to strings
    // For now, just return a placeholder
    return "Compute error";
}

// Print hardware capabilities
void compute_print_capabilities(const ComputeContext* ctx) {
    if (!ctx) return;
    
    printf("Hardware Capabilities:\n");
    printf("  CPU:\n");
    printf("    Cores: %d\n", ctx->hw_caps.num_cpu_cores);
    printf("    AVX-512: %s\n", ctx->hw_caps.has_avx512 ? "Yes" : "No");
    printf("    AVX2: %s\n", ctx->hw_caps.has_avx2 ? "Yes" : "No");
    printf("    SSE4: %s\n", ctx->hw_caps.has_sse4 ? "Yes" : "No");
    printf("    NEON: %s\n", ctx->hw_caps.has_neon ? "Yes" : "No");
    
    printf("  GPU:\n");
    printf("    CUDA available: %s\n", ctx->hw_caps.cuda_available ? "Yes" : "No");
    printf("    ROCm available: %s\n", ctx->hw_caps.rocm_available ? "Yes" : "No");
    printf("    Metal available: %s\n", ctx->hw_caps.metal_available ? "Yes" : "No");
    printf("    Number of devices: %d\n", ctx->hw_caps.num_gpu_devices);
    
    for (int i = 0; i < ctx->hw_caps.num_gpu_devices; i++) {
        printf("    Device %d memory: %zu MB\n", i, 
               ctx->hw_caps.gpu_memory[i] / (1024 * 1024));
    }
}

// Get device count
int compute_get_device_count(void) {
    // This should call into the GPU API to get the actual device count
    // For now, return a placeholder value
#ifdef ONEBIT_USE_CUDA
    return 1;  // Placeholder
#else
    return 0;
#endif
}

// Get device info
int compute_get_device_info(int device_id, char* name, size_t* memory) {
    if (device_id < 0 || device_id >= compute_get_device_count()) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // This should call into the GPU API to get the actual device info
    // For now, set placeholder values
    if (name) {
        strcpy(name, "NVIDIA GeForce RTX 3090");  // Placeholder
    }
    
    if (memory) {
        *memory = 24ULL * 1024 * 1024 * 1024;  // 24GB (placeholder)
    }
    
    return ONEBIT_SUCCESS;
} 