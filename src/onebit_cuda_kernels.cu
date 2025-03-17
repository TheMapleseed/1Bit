#include "onebit_compute.h"
#include "onebit_compute_types.h"
#include "onebit_cuda_kernels.h"

#define BLOCK_SIZE 16

// CUDA matrix multiplication kernel
__global__ void matmul_kernel(
    const float* A, const float* B, float* C,
    const size_t M, const size_t K, const size_t N) 
{
    // Block indices
    const size_t block_row = blockIdx.y;
    const size_t block_col = blockIdx.x;
    
    // Thread indices
    const size_t thread_row = threadIdx.y;
    const size_t thread_col = threadIdx.x;
    
    // Global indices
    const size_t row = block_row * BLOCK_SIZE + thread_row;
    const size_t col = block_col * BLOCK_SIZE + thread_col;
    
    // Shared memory for tiles
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (size_t t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * BLOCK_SIZE + thread_col < K) {
            As[thread_row][thread_col] = A[row * K + t * BLOCK_SIZE + thread_col];
        } else {
            As[thread_row][thread_col] = 0.0f;
        }
        
        if (t * BLOCK_SIZE + thread_row < K && col < N) {
            Bs[thread_row][thread_col] = B[(t * BLOCK_SIZE + thread_row) * N + col];
        } else {
            Bs[thread_row][thread_col] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (size_t k = 0; k < BLOCK_SIZE; k++) {
            sum += As[thread_row][k] * Bs[k][thread_col];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// CUDA wrapper function
void onebit_cuda_matmul(const Matrix* A, const Matrix* B, Matrix* C) {
    const size_t M = A->rows;
    const size_t K = A->cols;
    const size_t N = B->cols;
    
    // Grid and block dimensions
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    
    // Launch kernel
    matmul_kernel<<<grid, block>>>(
        (float*)A->data, (float*)B->data, (float*)C->data,
        M, K, N
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
}

// Add other CUDA kernels here... 