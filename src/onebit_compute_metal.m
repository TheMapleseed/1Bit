#import "onebit_compute_metal.h"
#include "onebit_error.h"
#include <simd/simd.h>

// Optimized Metal shader source with SIMD and tiling
static const char* metal_shader_source = R"(
#include <metal_stdlib>
using namespace metal;

kernel void matmul_optimized(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]])
{
    constexpr uint TILE_SIZE = 16;
    threadgroup float a_shared[TILE_SIZE][TILE_SIZE];
    threadgroup float b_shared[TILE_SIZE][TILE_SIZE];
    
    float acc = 0.0f;
    
    for (uint t = 0; t < K; t += TILE_SIZE) {
        // Load tiles collaboratively
        if (tid.x < TILE_SIZE && tid.y < TILE_SIZE) {
            uint row = bid.y * TILE_SIZE + tid.y;
            uint col = t + tid.x;
            if (row < M && col < K) {
                a_shared[tid.y][tid.x] = a[row * K + col];
            }
            
            row = t + tid.y;
            col = bid.x * TILE_SIZE + tid.x;
            if (row < K && col < N) {
                b_shared[tid.y][tid.x] = b[row * N + col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute tile product
        if (gid.x < N && gid.y < M) {
            for (uint k = 0; k < TILE_SIZE; k++) {
                acc += a_shared[tid.y][k] * b_shared[k][tid.x];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (gid.x < N && gid.y < M) {
        c[gid.y * N + gid.x] = acc;
    }
}
)";

int metal_init(ComputeContext* ctx) {
    MetalContext* metal_ctx = malloc(sizeof(MetalContext));
    if (!metal_ctx) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Initialize Metal device and command queue
    metal_ctx->device = MTLCreateSystemDefaultDevice();
    if (!metal_ctx->device) {
        free(metal_ctx);
        return ONEBIT_ERROR_INITIALIZATION;
    }
    
    metal_ctx->command_queue = [metal_ctx->device newCommandQueue];
    if (!metal_ctx->command_queue) {
        free(metal_ctx);
        return ONEBIT_ERROR_INITIALIZATION;
    }
    
    // Create compute pipeline
    NSError* error = nil;
    id<MTLLibrary> library = [metal_ctx->device newLibraryWithSource:
                             @(metal_shader_source) options:nil error:&error];
    if (!library) {
        free(metal_ctx);
        return ONEBIT_ERROR_INITIALIZATION;
    }
    
    metal_ctx->library = library;
    metal_ctx->capabilities = metal_get_capabilities(metal_ctx->device);
    
    ctx->device_ctx = metal_ctx;
    return ONEBIT_SUCCESS;
} 