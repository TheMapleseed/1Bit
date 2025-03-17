#include "onebit/onebit_cuda.h"
#include "onebit/onebit_error.h"
#include <string.h>

// CUDA kernel for optimized matrix multiplication
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    const int TILE_SIZE = 16;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        
        if (t * TILE_SIZE + threadIdx.y < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Initialize CUDA context
int cuda_init(OneBitCUDAContext* ctx, int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    ctx->device_id = device_id;
    ctx->available_memory = prop.totalGlobalMem;
    
    // Initialize cuBLAS
    CUDA_CHECK(cublasCreate(&ctx->cublas_handle));
    
    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&ctx->stream));
    
    // Allocate workspace
    size_t workspace_size = 1024 * 1024 * 1024;  // 1GB workspace
    CUDA_CHECK(cudaMalloc(&ctx->workspace, workspace_size));
    ctx->workspace_size = workspace_size;
    
    return ONEBIT_SUCCESS;
}

// Cleanup CUDA resources
void cuda_cleanup(OneBitCUDAContext* ctx) {
    if (ctx->workspace)
        cudaFree(ctx->workspace);
    
    if (ctx->stream)
        cudaStreamDestroy(ctx->stream);
    
    if (ctx->cublas_handle)
        cublasDestroy(ctx->cublas_handle);
}

// Get CUDA error string
const char* cuda_get_error_string(int error) {
    return cudaGetErrorString((cudaError_t)error);
} 