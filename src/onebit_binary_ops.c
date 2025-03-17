#include "onebit/onebit_ops.h"
#include "onebit/onebit_memory.h"
#include "onebit/onebit_error.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// SIMD includes for optimization
#ifdef __AVX512F__
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#endif

// Count number of bits set in a byte (popcount)
static inline int count_bits(uint8_t byte) {
    static const int lookup[16] = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    };
    return lookup[byte & 0x0F] + lookup[byte >> 4];
}

// Initialize a binary tensor
int binary_tensor_init(OneBitContext* ctx, BinaryTensor* tensor, 
                     const int64_t* shape, int ndim, bool requires_grad) {
    if (!ctx || !tensor || !shape || ndim <= 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Calculate total size
    size_t total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    
    // Calculate bit-packed size (1 bit per weight, 8 weights per byte)
    size_t packed_size = (total_size + 7) / 8; // Round up to nearest byte
    
    // Allocate memory
    tensor->data = (uint8_t*)onebit_malloc(ctx, packed_size);
    if (!tensor->data) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Allocate shape
    tensor->shape = (int64_t*)onebit_malloc(ctx, ndim * sizeof(int64_t));
    if (!tensor->shape) {
        onebit_free(ctx, tensor->data);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Allocate strides
    tensor->strides = (int64_t*)onebit_malloc(ctx, ndim * sizeof(int64_t));
    if (!tensor->strides) {
        onebit_free(ctx, tensor->data);
        onebit_free(ctx, tensor->shape);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Allocate scales (one per output channel for typical weight matrices)
    size_t output_dim = shape[0]; // Assume first dimension is output channels
    tensor->scales = (float*)onebit_malloc(ctx, output_dim * sizeof(float));
    if (!tensor->scales) {
        onebit_free(ctx, tensor->data);
        onebit_free(ctx, tensor->shape);
        onebit_free(ctx, tensor->strides);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Copy shape
    memcpy(tensor->shape, shape, ndim * sizeof(int64_t));
    tensor->ndim = ndim;
    
    // Compute strides (row-major layout)
    tensor->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        tensor->strides[i] = tensor->strides[i + 1] * shape[i + 1];
    }
    
    // Initialize data and scales
    memset(tensor->data, 0, packed_size);
    for (size_t i = 0; i < output_dim; i++) {
        tensor->scales[i] = 1.0f;
    }
    
    tensor->size = total_size;
    tensor->requires_grad = requires_grad;
    
    return ONEBIT_SUCCESS;
}

// Free a binary tensor
void binary_tensor_free(OneBitContext* ctx, BinaryTensor* tensor) {
    if (!ctx || !tensor) return;
    
    if (tensor->data) onebit_free(ctx, tensor->data);
    if (tensor->scales) onebit_free(ctx, tensor->scales);
    if (tensor->shape) onebit_free(ctx, tensor->shape);
    if (tensor->strides) onebit_free(ctx, tensor->strides);
    
    tensor->data = NULL;
    tensor->scales = NULL;
    tensor->shape = NULL;
    tensor->strides = NULL;
    tensor->size = 0;
    tensor->ndim = 0;
}

// Binarize weights (convert float weights to binary representation)
int binarize_weights(const float* weights, uint8_t* binary_weights,
                   float* scales, const BinaryQuantizeConfig* config,
                   const int64_t* dims, int ndim) {
    if (!weights || !binary_weights || !scales || !config || !dims || ndim <= 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Calculate dimensions
    size_t output_dim = dims[0]; // Assume first dimension is output features
    size_t input_dim = 1;
    for (int i = 1; i < ndim; i++) {
        input_dim *= dims[i];
    }
    
    // Zero out binary weights
    size_t packed_size = (output_dim * input_dim + 7) / 8;
    memset(binary_weights, 0, packed_size);
    
    // Process each output channel
    #pragma omp parallel for
    for (size_t out_idx = 0; out_idx < output_dim; out_idx++) {
        float sum_abs = 0.0f;
        float threshold = config->threshold;
        
        // Calculate per-channel threshold and scale if needed
        if (config->per_channel) {
            sum_abs = 0.0f;
            for (size_t in_idx = 0; in_idx < input_dim; in_idx++) {
                sum_abs += fabsf(weights[out_idx * input_dim + in_idx]);
            }
            
            // Set threshold as mean or use provided threshold
            if (config->threshold <= 0.0f) {
                threshold = sum_abs / input_dim;
            }
        }
        
        // Calculate scale factor
        scales[out_idx] = sum_abs / input_dim;
        if (scales[out_idx] < 1e-6f) {
            scales[out_idx] = 1e-6f; // Prevent division by zero
        }
        
        // Binarize weights
        for (size_t in_idx = 0; in_idx < input_dim; in_idx++) {
            size_t flat_idx = out_idx * input_dim + in_idx;
            size_t byte_idx = flat_idx / 8;
            size_t bit_idx = flat_idx % 8;
            
            float w = weights[flat_idx];
            bool bit_value;
            
            if (config->use_symmetric) {
                // -1/+1 binarization
                bit_value = w > 0.0f;
            } else {
                // 0/1 binarization
                bit_value = fabsf(w) > threshold;
            }
            
            // Set the bit
            if (bit_value) {
                binary_weights[byte_idx] |= (1 << bit_idx);
            }
        }
    }
    
    return ONEBIT_SUCCESS;
}

// Optimized binary matrix multiplication implementation
int compute_binary_matmul(const void* a, const void* b, void* c,
                        const int64_t* dims, const float* scales) {
    if (!a || !b || !c || !dims || !scales) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    const uint8_t* bin_a = (const uint8_t*)a;
    const float* b_data = (const float*)b;
    float* c_data = (float*)c;
    
    int64_t M = dims[0]; // Output rows
    int64_t N = dims[1]; // Output columns
    int64_t K = dims[2]; // Inner dimension
    
    // Calculate packed dimension (bits per byte)
    int64_t K_bytes = (K + 7) / 8;
    
    // Process each output element
    #pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            
            // Vectorized binary dot product
            #ifdef __AVX512F__
            // AVX-512 implementation
            // Process 512 binary weights at a time (64 bytes)
            int64_t k_block;
            for (k_block = 0; k_block + 512 <= K; k_block += 512) {
                // Implementation for AVX-512
                // This is a placeholder - actual implementation would use AVX-512 intrinsics
            }
            
            // Process remaining elements
            for (int64_t k_byte = k_block / 8; k_byte < K_bytes; k_byte++) {
                uint8_t byte_val = bin_a[i * K_bytes + k_byte];
                for (int bit = 0; bit < 8; bit++) {
                    int64_t k = k_byte * 8 + bit;
                    if (k < K) {
                        bool bit_val = (byte_val >> bit) & 1;
                        sum += bit_val ? b_data[k * N + j] : 0.0f;
                    }
                }
            }
            #else
            // Scalar implementation
            for (int64_t k_byte = 0; k_byte < K_bytes; k_byte++) {
                uint8_t byte_val = bin_a[i * K_bytes + k_byte];
                for (int bit = 0; bit < 8; bit++) {
                    int64_t k = k_byte * 8 + bit;
                    if (k < K) {
                        bool bit_val = (byte_val >> bit) & 1;
                        sum += bit_val ? b_data[k * N + j] : 0.0f;
                    }
                }
            }
            #endif
            
            // Apply scale
            c_data[i * N + j] = sum * scales[i];
        }
    }
    
    return ONEBIT_SUCCESS;
}

// XNOR operation for binary tensors (efficient dot product when both a and b are binary)
int binary_xnor_popcount(const uint8_t* a, const uint8_t* b, 
                       size_t length, float* result) {
    if (!a || !b || !result) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    size_t num_bytes = (length + 7) / 8;
    int total_bits = 0;
    
    // Loop through bytes and count matching bits using XNOR (~(a^b))
    for (size_t i = 0; i < num_bytes; i++) {
        uint8_t xnor_result = ~(a[i] ^ b[i]);
        total_bits += count_bits(xnor_result);
    }
    
    // Convert bit count to dot product result (2*popcount - length)
    *result = (2.0f * total_bits - length);
    
    return ONEBIT_SUCCESS;
}

// Binary dot product (when one tensor is binary and one is float)
int binary_dot_product(const uint8_t* a, const float* b, float* c,
                     const int64_t* dims, const float* scales) {
    if (!a || !b || !c || !dims || !scales) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Extract dimensions
    int64_t batch_size = dims[0];
    int64_t vec_len = dims[1];
    
    // Calculate packed dimension (bits per byte)
    int64_t vec_bytes = (vec_len + 7) / 8;
    
    // Process each batch element
    #pragma omp parallel for
    for (int64_t i = 0; i < batch_size; i++) {
        float sum = 0.0f;
        
        // Process each byte
        for (int64_t byte_idx = 0; byte_idx < vec_bytes; byte_idx++) {
            uint8_t byte_val = a[i * vec_bytes + byte_idx];
            
            // Process each bit in the byte
            for (int bit = 0; bit < 8; bit++) {
                int64_t idx = byte_idx * 8 + bit;
                if (idx < vec_len) {
                    bool bit_val = (byte_val >> bit) & 1;
                    sum += bit_val ? b[i * vec_len + idx] : 0.0f;
                }
            }
        }
        
        // Store result with scaling
        c[i] = sum * scales[i % dims[2]]; // dims[2] = number of output channels
    }
    
    return ONEBIT_SUCCESS;
}

// Apply binarization to an existing tensor
int tensor_binarize(OneBitContext* ctx, const Tensor* input_tensor, 
                  BinaryTensor* output_tensor,
                  const BinaryQuantizeConfig* config) {
    if (!ctx || !input_tensor || !output_tensor || !config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Initialize binary tensor
    int result = binary_tensor_init(ctx, output_tensor, 
                                 input_tensor->shape, input_tensor->ndim,
                                 input_tensor->requires_grad);
    if (result != ONEBIT_SUCCESS) {
        return result;
    }
    
    // Perform binarization
    result = binarize_weights((const float*)input_tensor->data, 
                           output_tensor->data,
                           output_tensor->scales, 
                           config, 
                           input_tensor->shape, 
                           input_tensor->ndim);
    
    if (result != ONEBIT_SUCCESS) {
        binary_tensor_free(ctx, output_tensor);
        return result;
    }
    
    return ONEBIT_SUCCESS;
}

// Apply GELU activation
int compute_gelu(const float* input, float* output, size_t size) {
    if (!input || !output) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/π)
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        float x = input[i];
        float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
        output[i] = 0.5f * x * (1.0f + tanhf(tanh_arg));
    }
    
    return ONEBIT_SUCCESS;
} 