#import "../test_utils.h"
#import <Metal/Metal.h>
#import <pthread.h>

// Metal Memory allocation tracking
typedef struct {
    id<MTLBuffer> buffer;
    size_t size;
    const char* file;
    int line;
    MTLResourceOptions options;
} MetalMemoryAllocation;

#define MAX_METAL_ALLOCATIONS 2000
static MetalMemoryAllocation g_metal_allocations[MAX_METAL_ALLOCATIONS];
static size_t g_num_metal_allocations = 0;
static size_t g_total_metal_allocated = 0;
static pthread_mutex_t g_metal_mutex = PTHREAD_MUTEX_INITIALIZER;

// Metal Memory tracking functions
id<MTLBuffer> tracked_metal_buffer(id<MTLDevice> device, 
                                 size_t size,
                                 MTLResourceOptions options,
                                 const char* file,
                                 int line) {
    id<MTLBuffer> buffer = [device newBufferWithLength:size
                                             options:options];
    
    if (buffer) {
        pthread_mutex_lock(&g_metal_mutex);
        
        if (g_num_metal_allocations < MAX_METAL_ALLOCATIONS) {
            g_metal_allocations[g_num_metal_allocations++] = (MetalMemoryAllocation){
                .buffer = buffer,
                .size = size,
                .file = file,
                .line = line,
                .options = options
            };
            g_total_metal_allocated += size;
        }
        
        pthread_mutex_unlock(&g_metal_mutex);
    }
    return buffer;
}

void tracked_metal_release(id<MTLBuffer> buffer) {
    if (!buffer) return;
    
    pthread_mutex_lock(&g_metal_mutex);
    
    for (size_t i = 0; i < g_num_metal_allocations; i++) {
        if (g_metal_allocations[i].buffer == buffer) {
            g_total_metal_allocated -= g_metal_allocations[i].size;
            memmove(&g_metal_allocations[i], &g_metal_allocations[i + 1],
                   (g_num_metal_allocations - i - 1) * sizeof(MetalMemoryAllocation));
            g_num_metal_allocations--;
            break;
        }
    }
    
    pthread_mutex_unlock(&g_metal_mutex);
    [buffer release];
} 