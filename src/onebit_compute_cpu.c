#include "onebit/onebit_compute.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

// Thread pool configuration
#define MAX_THREADS 32
#define MIN_THREADS 2

typedef struct {
    pthread_t thread;
    bool active;
    void (*task)(void*);
    void* args;
} ThreadInfo;

typedef struct {
    ThreadInfo threads[MAX_THREADS];
    size_t num_threads;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    bool shutdown;
    
    // CPU capabilities
    bool has_avx2;
    bool has_avx512;
    int cache_line_size;
} CPUContext;

static CPUContext* g_cpu_context = NULL;

// Thread worker function
static void* thread_worker(void* arg) {
    ThreadInfo* info = (ThreadInfo*)arg;
    
    while (true) {
        pthread_mutex_lock(&g_cpu_context->mutex);
        
        while (!info->task && !g_cpu_context->shutdown) {
            pthread_cond_wait(&g_cpu_context->condition, 
                            &g_cpu_context->mutex);
        }
        
        if (g_cpu_context->shutdown) {
            pthread_mutex_unlock(&g_cpu_context->mutex);
            break;
        }
        
        // Get task
        void (*task)(void*) = info->task;
        void* task_args = info->args;
        info->task = NULL;
        
        pthread_mutex_unlock(&g_cpu_context->mutex);
        
        // Execute task
        if (task) {
            task(task_args);
        }
    }
    
    return NULL;
}

// CPU capability detection
static void detect_cpu_capabilities(CPUContext* ctx) {
    #if defined(__x86_64__) || defined(_M_X64)
    int cpu_info[4];
    
    // Check AVX2
    __cpuid(cpu_info, 7);
    ctx->has_avx2 = (cpu_info[1] & (1 << 5)) != 0;
    
    // Check AVX512
    ctx->has_avx512 = (cpu_info[1] & (1 << 16)) != 0;
    
    // Get cache line size
    __cpuid(cpu_info, 1);
    ctx->cache_line_size = ((cpu_info[1] >> 8) & 0xff) * 8;
    #else
    ctx->has_avx2 = false;
    ctx->has_avx512 = false;
    ctx->cache_line_size = 64;  // Safe default
    #endif
}

// Initialize CPU context
int cpu_init_context(void) {
    if (g_cpu_context) {
        return 0;  // Already initialized
    }
    
    g_cpu_context = calloc(1, sizeof(CPUContext));
    if (!g_cpu_context) {
        return -1;
    }
    
    // Initialize mutex and condition
    pthread_mutex_init(&g_cpu_context->mutex, NULL);
    pthread_cond_init(&g_cpu_context->condition, NULL);
    
    // Detect CPU capabilities
    detect_cpu_capabilities(g_cpu_context);
    
    // Determine number of threads
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    g_cpu_context->num_threads = num_cores > MAX_THREADS ? 
                                MAX_THREADS : 
                                (num_cores < MIN_THREADS ? MIN_THREADS : num_cores);
    
    // Create worker threads
    for (size_t i = 0; i < g_cpu_context->num_threads; i++) {
        ThreadInfo* info = &g_cpu_context->threads[i];
        info->active = true;
        info->task = NULL;
        info->args = NULL;
        
        if (pthread_create(&info->thread, NULL, thread_worker, info) != 0) {
            // Cleanup on failure
            for (size_t j = 0; j < i; j++) {
                g_cpu_context->threads[j].active = false;
                pthread_join(g_cpu_context->threads[j].thread, NULL);
            }
            pthread_mutex_destroy(&g_cpu_context->mutex);
            pthread_cond_destroy(&g_cpu_context->condition);
            free(g_cpu_context);
            g_cpu_context = NULL;
            return -1;
        }
    }
    
    return 0;
}

// Submit task to thread pool
int cpu_submit_task(void (*task)(void*), void* args) {
    if (!g_cpu_context || !task) {
        return -1;
    }
    
    pthread_mutex_lock(&g_cpu_context->mutex);
    
    // Find available thread
    bool task_assigned = false;
    for (size_t i = 0; i < g_cpu_context->num_threads; i++) {
        if (!g_cpu_context->threads[i].task) {
            g_cpu_context->threads[i].task = task;
            g_cpu_context->threads[i].args = args;
            task_assigned = true;
            break;
        }
    }
    
    pthread_cond_broadcast(&g_cpu_context->condition);
    pthread_mutex_unlock(&g_cpu_context->mutex);
    
    return task_assigned ? 0 : -1;
}

// Cleanup CPU context
void cpu_cleanup(void) {
    if (!g_cpu_context) {
        return;
    }
    
    // Signal shutdown
    pthread_mutex_lock(&g_cpu_context->mutex);
    g_cpu_context->shutdown = true;
    pthread_cond_broadcast(&g_cpu_context->condition);
    pthread_mutex_unlock(&g_cpu_context->mutex);
    
    // Wait for threads to finish
    for (size_t i = 0; i < g_cpu_context->num_threads; i++) {
        if (g_cpu_context->threads[i].active) {
            pthread_join(g_cpu_context->threads[i].thread, NULL);
        }
    }
    
    // Cleanup resources
    pthread_mutex_destroy(&g_cpu_context->mutex);
    pthread_cond_destroy(&g_cpu_context->condition);
    free(g_cpu_context);
    g_cpu_context = NULL;
}

// Export CPU kernels for dynamic loading
void* get_kernel_cpu_matmul() {
    return (void*)cpu_matmul_kernel;
}

// ... other kernel exports