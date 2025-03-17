#ifndef ONEBIT_COMPUTE_METAL_H
#define ONEBIT_COMPUTE_METAL_H

#include "onebit_compute.h"
#ifdef __APPLE__
#import <Metal/Metal.h>

// Metal-specific compute capabilities
#define METAL_CAP_SIMD        (1 << 8)
#define METAL_CAP_UNIFIED_MEM (1 << 9)
#define METAL_CAP_TILE_SHADER (1 << 10)

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;
    id<MTLLibrary> library;
    struct {
        id<MTLComputePipelineState> attention;
        id<MTLComputePipelineState> layernorm;
        id<MTLComputePipelineState> matmul;
        id<MTLComputePipelineState> softmax;
        id<MTLComputePipelineState> gelu;
    } pipelines;
    id<MTLBuffer> workspace;
    size_t workspace_size;
    dispatch_semaphore_t completion_fence;
    NSMutableArray* command_buffers;
    pthread_mutex_t device_mutex;
    uint32_t capabilities;
} MetalContext;

// Function declarations
int metal_init(ComputeContext* ctx);
void metal_cleanup(ComputeContext* ctx);
int metal_synchronize(MetalContext* ctx);
bool metal_is_available(void);
uint32_t metal_get_capabilities(id<MTLDevice> device);

#endif // __APPLE__
#endif // ONEBIT_COMPUTE_METAL_H 