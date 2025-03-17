#include <metal_stdlib>
using namespace metal;

// Constants
constant int BLOCK_SIZE = 16;

// Matrix multiplication kernel
kernel void onebit_matmul(
    const device float* A [[buffer(0)]],
    const device float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]])
{
    // Shared memory for tiles
    threadgroup float As[BLOCK_SIZE][BLOCK_SIZE];
    threadgroup float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    const uint row = bid.y * BLOCK_SIZE + tid.y;
    const uint col = bid.x * BLOCK_SIZE + tid.x;
    
    float sum = 0.0;
    
    // Loop over tiles
    for (uint t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * BLOCK_SIZE + tid.x < K) {
            As[tid.y][tid.x] = A[row * K + t * BLOCK_SIZE + tid.x];
        } else {
            As[tid.y][tid.x] = 0.0;
        }
        
        if (t * BLOCK_SIZE + tid.y < K && col < N) {
            Bs[tid.y][tid.x] = B[(t * BLOCK_SIZE + tid.y) * N + col];
        } else {
            Bs[tid.y][tid.x] = 0.0;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (uint k = 0; k < BLOCK_SIZE; k++) {
            sum += As[tid.y][k] * Bs[k][tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Add other Metal kernels here... 