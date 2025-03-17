#include "../test_utils.h"
#include "onebit_compute.h"
#include <cuda_runtime.h>
#include <pthread.h>

// GPU Memory allocation tracking
typedef struct {
    void* ptr;
    size_t size;
    const char* file;
    int line;
    bool is_gpu;
    int device_id;
} GPUMemoryAllocation;

#define MAX_GPU_ALLOCATIONS 2000
static GPUMemoryAllocation g_gpu_allocations[MAX_GPU_ALLOCATIONS];
static size_t g_num_gpu_allocations = 0;
static size_t g_total_gpu_allocated = 0;
static pthread_mutex_t g_gpu_mutex = PTHREAD_MUTEX_INITIALIZER;

// GPU Memory tracking functions
void* tracked_cuda_malloc(size_t size, const char* file, int line) {
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    
    if (err == cudaSuccess && ptr) {
        pthread_mutex_lock(&g_gpu_mutex);
        
        if (g_num_gpu_allocations < MAX_GPU_ALLOCATIONS) {
            int device_id;
            cudaGetDevice(&device_id);
            
            g_gpu_allocations[g_num_gpu_allocations++] = (GPUMemoryAllocation){
                .ptr = ptr,
                .size = size,
                .file = file,
                .line = line,
                .is_gpu = true,
                .device_id = device_id
            };
            g_total_gpu_allocated += size;
        }
        
        pthread_mutex_unlock(&g_gpu_mutex);
    }
    return ptr;
}

void tracked_cuda_free(void* ptr) {
    if (!ptr) return;
    
    pthread_mutex_lock(&g_gpu_mutex);
    
    for (size_t i = 0; i < g_num_gpu_allocations; i++) {
        if (g_gpu_allocations[i].ptr == ptr) {
            g_total_gpu_allocated -= g_gpu_allocations[i].size;
            memmove(&g_gpu_allocations[i], &g_gpu_allocations[i + 1],
                   (g_num_gpu_allocations - i - 1) * sizeof(GPUMemoryAllocation));
            g_num_gpu_allocations--;
            break;
        }
    }
    
    pthread_mutex_unlock(&g_gpu_mutex);
    cudaFree(ptr);
}

// Thread-safe memory tracking for both CPU and GPU
static pthread_mutex_t g_mem_mutex = PTHREAD_MUTEX_INITIALIZER;

void* tracked_malloc_ts(size_t size, const char* file, int line) {
    pthread_mutex_lock(&g_mem_mutex);
    void* ptr = tracked_malloc(size, file, line);
    pthread_mutex_unlock(&g_mem_mutex);
    return ptr;
}

void tracked_free_ts(void* ptr) {
    pthread_mutex_lock(&g_mem_mutex);
    tracked_free(ptr);
    pthread_mutex_unlock(&g_mem_mutex);
}

// GPU Memory leak test cases
void test_gpu_memory_leaks(TestContext* test) {
    pthread_mutex_lock(&g_gpu_mutex);
    g_num_gpu_allocations = 0;
    g_total_gpu_allocated = 0;
    pthread_mutex_unlock(&g_gpu_mutex);
    
    ComputeContext* ctx;
    compute_init(&ctx, COMPUTE_DEVICE_CUDA);
    
    // Test GPU allocations
    const size_t size = 1024 * 1024;  // 1MB
    void* d_ptr = tracked_cuda_malloc(size, __FILE__, __LINE__);
    
    pthread_mutex_lock(&g_gpu_mutex);
    size_t after_alloc = g_total_gpu_allocated;
    pthread_mutex_unlock(&g_gpu_mutex);
    
    assert_test(test, "GPU Memory Allocation",
                after_alloc == size && d_ptr != NULL);
    
    tracked_cuda_free(d_ptr);
    
    pthread_mutex_lock(&g_gpu_mutex);
    assert_test(test, "GPU Memory Cleanup",
                g_total_gpu_allocated == 0 && g_num_gpu_allocations == 0);
    pthread_mutex_unlock(&g_gpu_mutex);
    
    compute_cleanup(ctx);
}

// Print both CPU and GPU memory leaks
void print_all_memory_leaks(void) {
    pthread_mutex_lock(&g_mem_mutex);
    pthread_mutex_lock(&g_gpu_mutex);
    
    if (g_num_allocations > 0 || g_num_gpu_allocations > 0) {
        printf("\nMemory Leaks Detected:\n");
        printf("=====================\n");
        
        // CPU leaks
        for (size_t i = 0; i < g_num_allocations; i++) {
            printf("CPU Leak: %zu bytes at %p (%s:%d)\n",
                   g_allocations[i].size,
                   g_allocations[i].ptr,
                   g_allocations[i].file,
                   g_allocations[i].line);
        }
        
        // GPU leaks
        for (size_t i = 0; i < g_num_gpu_allocations; i++) {
            printf("GPU Leak: %zu bytes at %p (Device %d, %s:%d)\n",
                   g_gpu_allocations[i].size,
                   g_gpu_allocations[i].ptr,
                   g_gpu_allocations[i].device_id,
                   g_gpu_allocations[i].file,
                   g_gpu_allocations[i].line);
        }
        
        printf("Total CPU leaked: %zu bytes\n", g_total_allocated);
        printf("Total GPU leaked: %zu bytes\n", g_total_gpu_allocated);
    }
    
    pthread_mutex_unlock(&g_gpu_mutex);
    pthread_mutex_unlock(&g_mem_mutex);
} 