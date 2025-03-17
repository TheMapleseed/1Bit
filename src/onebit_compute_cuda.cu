#include "onebit_compute_cuda.h"
#include "onebit_error.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Attention kernel matching preset implementation
__global__ void attention_kernel(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    const int batch_size,
    const int num_heads,
    const int seq_length,
    const int head_dim
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int i = threadIdx.x;
    
    __shared__ float scores[1024];  // Assuming max seq length of 1024
    
    if (b < batch_size && h < num_heads && i < seq_length) {
        // Compute attention scores
        float sum = 0.0f;
        for (int j = 0; j < seq_length; j++) {
            const float q = query[b * num_heads * seq_length * head_dim + 
                                h * seq_length * head_dim +
                                i * head_dim];
            const float k = key[b * num_heads * seq_length * head_dim +
                               h * seq_length * head_dim +
                               j * head_dim];
            
            // Dot product
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q * k;
            }
            score /= sqrtf(head_dim);
            
            scores[j] = score;
            sum += expf(score);
        }
        
        // Softmax normalization
        for (int j = 0; j < seq_length; j++) {
            scores[j] = expf(scores[j]) / sum;
        }
        
        // Compute weighted sum of values
        for (int d = 0; d < head_dim; d++) {
            float weighted_sum = 0.0f;
            for (int j = 0; j < seq_length; j++) {
                weighted_sum += scores[j] * value[b * num_heads * seq_length * head_dim +
                                                h * seq_length * head_dim +
                                                j * head_dim + d];
            }
            output[b * num_heads * seq_length * head_dim +
                  h * seq_length * head_dim +
                  i * head_dim + d] = weighted_sum;
        }
    }
}

// Layer normalization kernel
__global__ void layernorm_kernel(
    float* input,
    const float* gamma,
    const float* beta,
    const int size,
    const float eps
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        __shared__ float mean, variance;
        
        // Compute mean
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += input[i];
        }
        mean = sum / size;
        
        // Compute variance
        sum = 0.0f;
        for (int i = 0; i < size; i++) {
            float diff = input[i] - mean;
            sum += diff * diff;
        }
        variance = sum / size;
        
        // Normalize
        input[idx] = gamma[idx] * (input[idx] - mean) / sqrtf(variance + eps) + beta[idx];
    }
}

// Matrix multiplication kernel
__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    const int M,
    const int N,
    const int K
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled matrix multiplication implementation
#define TILE_SIZE 16

__global__ void matmul_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int bx = blockIdx.x * TILE_SIZE;
    const int by = blockIdx.y * TILE_SIZE;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int row = by + ty;
    const int col = bx + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles cooperatively
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (t * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial products
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Batched matrix multiplication
__global__ void batched_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int batch_size,
    const int M, const int N, const int K
) {
    const int batch_idx = blockIdx.z;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && row < M && col < N) {
        const float* batch_A = A + batch_idx * M * K;
        const float* batch_B = B + batch_idx * K * N;
        float* batch_C = C + batch_idx * M * N;
        
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += batch_A[row * K + k] * batch_B[k * N + col];
        }
        batch_C[row * N + col] = sum;
    }
}

// Optimized attention mechanism kernels

// QK Computation kernel
__global__ void qk_attention_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    float* __restrict__ scores,
    const int batch_size,
    const int num_heads,
    const int seq_length,
    const int head_dim,
    const float scale
) {
    __shared__ float q_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float k_shared[TILE_SIZE][TILE_SIZE];
    
    const int b = blockIdx.z;  // batch index
    const int h = blockIdx.y;  // head index
    const int row = blockIdx.x * TILE_SIZE + threadIdx.y;
    const int col = threadIdx.x;
    
    const int batch_offset = b * num_heads * seq_length * head_dim;
    const int head_offset = h * seq_length * head_dim;
    
    float qk_sum = 0.0f;
    
    // Tile-based computation
    for (int tile = 0; tile < (head_dim + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load query and key tiles
        if (row < seq_length && tile * TILE_SIZE + col < head_dim) {
            q_shared[threadIdx.y][threadIdx.x] = query[
                batch_offset + head_offset + 
                row * head_dim + tile * TILE_SIZE + col
            ];
            k_shared[threadIdx.y][threadIdx.x] = key[
                batch_offset + head_offset + 
                col * head_dim + tile * TILE_SIZE + threadIdx.y
            ];
        } else {
            q_shared[threadIdx.y][threadIdx.x] = 0.0f;
            k_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot products
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            qk_sum += q_shared[threadIdx.y][k] * k_shared[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Apply scaling and write result
    if (row < seq_length && col < seq_length) {
        scores[batch_offset + head_offset + row * seq_length + col] = 
            qk_sum * scale;
    }
}

// Softmax kernel for attention scores
__global__ void attention_softmax_kernel(
    float* __restrict__ scores,
    const int batch_size,
    const int num_heads,
    const int seq_length
) {
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int row = blockIdx.x;
    
    const int offset = b * num_heads * seq_length * seq_length +
                      h * seq_length * seq_length +
                      row * seq_length;
    
    if (row >= seq_length) return;
    
    // Find max for numerical stability
    float max_val = scores[offset];
    for (int i = 1; i < seq_length; i++) {
        max_val = max(max_val, scores[offset + i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < seq_length; i++) {
        scores[offset + i] = __expf(scores[offset + i] - max_val);
        sum += scores[offset + i];
    }
    
    // Normalize
    float inv_sum = __fdividef(1.0f, sum);
    for (int i = 0; i < seq_length; i++) {
        scores[offset + i] *= inv_sum;
    }
}

// Attention output computation kernel
__global__ void attention_output_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ value,
    float* __restrict__ output,
    const int batch_size,
    const int num_heads,
    const int seq_length,
    const int head_dim
) {
    __shared__ float s_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float v_shared[TILE_SIZE][TILE_SIZE];
    
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int row = blockIdx.x * TILE_SIZE + threadIdx.y;
    const int col = threadIdx.x;
    
    const int batch_head_offset = b * num_heads * seq_length * head_dim +
                                 h * seq_length * head_dim;
    
    float out_sum = 0.0f;
    
    // Tiled matrix multiplication
    for (int tile = 0; tile < (seq_length + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        if (row < seq_length && tile * TILE_SIZE + col < seq_length) {
            s_shared[threadIdx.y][threadIdx.x] = scores[
                batch_head_offset + row * seq_length + tile * TILE_SIZE + col
            ];
            v_shared[threadIdx.y][threadIdx.x] = value[
                batch_head_offset + (tile * TILE_SIZE + threadIdx.y) * head_dim + col
            ];
        } else {
            s_shared[threadIdx.y][threadIdx.x] = 0.0f;
            v_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            out_sum += s_shared[threadIdx.y][k] * v_shared[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < seq_length && col < head_dim) {
        output[batch_head_offset + row * head_dim + col] = out_sum;
    }
} 

extern "C" {
    // Export kernel functions for dynamic loading
    __host__ void* get_kernel_qk_attention() {
        return (void*)qk_attention_kernel;
    }
    
    __host__ void* get_kernel_softmax() {
        return (void*)attention_softmax_kernel;
    }
    
    // ... other kernel exports
} 