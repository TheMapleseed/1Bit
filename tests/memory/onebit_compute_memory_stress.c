#include "../test_utils.h"
#include "onebit_compute.h"
#include <pthread.h>
#include <unistd.h>

#define NUM_THREADS 8
#define ITERATIONS_PER_THREAD 1000
#define MAX_ALLOCATION_SIZE (1024 * 1024)  // 1MB

typedef struct {
    int thread_id;
    TestContext* test;
    ComputeContext* ctx;
    ComputeDeviceType device_type;
} ThreadArgs;

// Random size generator
static size_t random_size(void) {
    return (rand() % MAX_ALLOCATION_SIZE) + 1;
}

// Thread worker function
void* memory_stress_worker(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    void* allocations[100] = {NULL};  // Track recent allocations
    int alloc_index = 0;
    
    for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
        // Randomly choose between allocation and deallocation
        if (rand() % 2 == 0) {
            size_t size = random_size();
            void* ptr = NULL;
            
            switch (args->device_type) {
                case COMPUTE_DEVICE_CPU:
                    ptr = tracked_malloc_ts(size, __FILE__, __LINE__);
                    break;
                    
                case COMPUTE_DEVICE_CUDA:
                    ptr = tracked_cuda_malloc(size, __FILE__, __LINE__);
                    break;
                    
                case COMPUTE_DEVICE_METAL:
                    #ifdef __APPLE__
                    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
                    ptr = (__bridge void*)tracked_metal_buffer(
                        device, size, MTLResourceStorageModeShared,
                        __FILE__, __LINE__
                    );
                    #endif
                    break;
            }
            
            if (ptr) {
                if (allocations[alloc_index]) {
                    // Free previous allocation at this index
                    switch (args->device_type) {
                        case COMPUTE_DEVICE_CPU:
                            tracked_free_ts(allocations[alloc_index]);
                            break;
                        case COMPUTE_DEVICE_CUDA:
                            tracked_cuda_free(allocations[alloc_index]);
                            break;
                        case COMPUTE_DEVICE_METAL:
                            #ifdef __APPLE__
                            tracked_metal_release((__bridge id<MTLBuffer>)allocations[alloc_index]);
                            #endif
                            break;
                    }
                }
                allocations[alloc_index] = ptr;
                alloc_index = (alloc_index + 1) % 100;
            }
        } else if (alloc_index > 0) {
            // Deallocation
            int index = rand() % alloc_index;
            if (allocations[index]) {
                switch (args->device_type) {
                    case COMPUTE_DEVICE_CPU:
                        tracked_free_ts(allocations[index]);
                        break;
                    case COMPUTE_DEVICE_CUDA:
                        tracked_cuda_free(allocations[index]);
                        break;
                    case COMPUTE_DEVICE_METAL:
                        #ifdef __APPLE__
                        tracked_metal_release((__bridge id<MTLBuffer>)allocations[index]);
                        #endif
                        break;
                }
                allocations[index] = NULL;
            }
        }
        
        // Occasional compute operation
        if (i % 100 == 0) {
            Matrix mat_a = {0}; // Initialize test matrices
            Matrix mat_b = {0};
            Matrix mat_c = {0};
            struct { Matrix a, b, c; } params = {mat_a, mat_b, mat_c};
            compute_execute(args->ctx, "matrix_multiply", &params);
        }
        
        // Small sleep to allow other threads to run
        usleep(1);
    }
    
    // Cleanup remaining allocations
    for (int i = 0; i < 100; i++) {
        if (allocations[i]) {
            switch (args->device_type) {
                case COMPUTE_DEVICE_CPU:
                    tracked_free_ts(allocations[i]);
                    break;
                case COMPUTE_DEVICE_CUDA:
                    tracked_cuda_free(allocations[i]);
                    break;
                case COMPUTE_DEVICE_METAL:
                    #ifdef __APPLE__
                    tracked_metal_release((__bridge id<MTLBuffer>)allocations[i]);
                    #endif
                    break;
            }
        }
    }
    
    return NULL;
}

// Multi-threaded stress test
void test_memory_stress_threaded(TestContext* test) {
    pthread_t threads[NUM_THREADS];
    ThreadArgs thread_args[NUM_THREADS];
    
    ComputeDeviceType devices[] = {
        COMPUTE_DEVICE_CPU,
        COMPUTE_DEVICE_CUDA,
        COMPUTE_DEVICE_METAL
    };
    
    for (int device_idx = 0; device_idx < 3; device_idx++) {
        ComputeContext* ctx;
        compute_init(&ctx, devices[device_idx]);
        
        printf("\nRunning stress test for device: %d\n", devices[device_idx]);
        
        // Create threads
        for (int i = 0; i < NUM_THREADS; i++) {
            thread_args[i].thread_id = i;
            thread_args[i].test = test;
            thread_args[i].ctx = ctx;
            thread_args[i].device_type = devices[device_idx];
            
            pthread_create(&threads[i], NULL, memory_stress_worker, &thread_args[i]);
        }
        
        // Wait for threads to complete
        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(threads[i], NULL);
        }
        
        // Verify no leaks
        print_all_memory_leaks();
        assert_test(test, "Memory Stress Test - No Leaks",
                   g_total_allocated == 0 && 
                   g_total_gpu_allocated == 0 && 
                   g_total_metal_allocated == 0);
        
        compute_cleanup(ctx);
    }
} 