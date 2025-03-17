/**
 * @file onebit_ops.h
 * @brief Core operations for the OneBit library with binary optimization
 */

#ifndef ONEBIT_onebit_ops_H
#define ONEBIT_onebit_ops_H

#include "onebit_error.h"
#include <stdint.h>
#include <stdbool.h>

// Binary quantization configuration
typedef struct {
    float threshold;        // Threshold for binarization
    float scale_factor;     // Scaling factor for quantized values
    bool use_symmetric;     // Whether to use symmetric quantization
    bool per_channel;       // Whether to apply quantization per-channel
} BinaryQuantizeConfig;

// Binary tensor representation
typedef struct {
    uint8_t* data;          // Bit-packed data (1 bit per weight)
    float* scales;          // Scaling factors
    int64_t* shape;         // Tensor shape
    int ndim;               // Number of dimensions
    int64_t* strides;       // Strides for each dimension
    size_t size;            // Total number of elements
    bool requires_grad;     // Whether gradients are required for this tensor
} BinaryTensor;

/**
 * @brief Perform matrix multiplication C = A * B
 * 
 * @param a Pointer to matrix A
 * @param b Pointer to matrix B
 * @param c Pointer to result matrix C
 * @param dims Array with dimensions [M, N, K] where C is MxN, A is MxK, B is KxN
 * @return int Error code
 */
int compute_matmul(const void* a, const void* b, void* c, const int64_t* dims);

/**
 * @brief Perform matrix multiplication with binary weights C = A * B
 * 
 * @param a Pointer to binary matrix A
 * @param b Pointer to matrix B
 * @param c Pointer to result matrix C
 * @param dims Array with dimensions [M, N, K] where C is MxN, A is MxK, B is KxN
 * @param scales Scaling factors for binary values
 * @return int Error code
 */
int compute_binary_matmul(const void* a, const void* b, void* c, 
                        const int64_t* dims, const float* scales);

/**
 * @brief Apply layer normalization
 * 
 * @param input Input tensor
 * @param output Output normalized tensor
 * @param dims Array with dimensions [batch_size, hidden_size]
 * @return int Error code
 */
int compute_layernorm(const void* input, void* output, const int64_t* dims);

/**
 * @brief Apply softmax activation
 * 
 * @param input Input tensor
 * @param output Output tensor after softmax
 * @param dims Array with dimensions [batch_size, seq_length]
 * @return int Error code
 */
int compute_softmax(const void* input, void* output, const int64_t* dims);

/**
 * @brief Binarize weights for efficient computation
 * 
 * @param weights Input float weights
 * @param binary_weights Output binary weights
 * @param config Quantization configuration
 * @param dims Array with tensor dimensions
 * @return int Error code
 */
int binarize_weights(const float* weights, uint8_t* binary_weights,
                   float* scales, const BinaryQuantizeConfig* config,
                   const int64_t* dims, int ndim);

/**
 * @brief Apply GELU activation
 * 
 * @param input Input tensor
 * @param output Output tensor after GELU
 * @param size Number of elements
 * @return int Error code
 */
int compute_gelu(const float* input, float* output, size_t size);

/**
 * @brief Apply binary dot product with SIMD acceleration
 * 
 * @param a Binary tensor
 * @param b Float tensor
 * @param c Output tensor
 * @param dims Dimensions
 * @param scales Scaling factors
 * @return int Error code
 */
int binary_dot_product(const uint8_t* a, const float* b, float* c,
                     const int64_t* dims, const float* scales);

#endif // ONEBIT_OPS_H 