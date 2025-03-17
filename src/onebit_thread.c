#include "onebit/onebit_thread.h"
#include "onebit/onebit_error.h"
#include <stdlib.h>
#include <string.h>

// Worker thread function
static void* thread_worker(void* arg) {
    ThreadPoolContext* ctx = (ThreadPoolContext*)arg;
    ThreadTask* task = NULL;
    
    while (1) {
        pthread_mutex_lock(&ctx->mutex);
        
        // Wait for task or stop signal
        while (!ctx->should_stop && !ctx->task_queue) {
            pthread_cond_wait(&ctx->cond, &ctx->mutex);
        }
        
        if (ctx->should_stop && !ctx->task_queue) {
            pthread_mutex_unlock(&ctx->mutex);
            break;
        }
        
        // Get task from queue
        task = ctx->task_queue;
        if (task) {
            ctx->task_queue = task->next;
            if (!ctx->task_queue) {
                ctx->task_tail = NULL;
            }
            ctx->queue_size--;
            ctx->active_threads++;
        }
        
        pthread_mutex_unlock(&ctx->mutex);
        
        // Execute task
        if (task) {
            task->func(task->arg);
            free(task);
            
            pthread_mutex_lock(&ctx->mutex);
            ctx->active_threads--;
            pthread_mutex_unlock(&ctx->mutex);
        }
    }
    
    return NULL;
}

int thread_pool_init(ThreadPoolContext* ctx, const ThreadPoolConfig* config) {
    if (!ctx || !config || config->num_threads == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Initialize context
    memset(ctx, 0, sizeof(ThreadPoolContext));
    ctx->num_threads = config->num_threads;
    ctx->max_queue_size = config->queue_size;
    ctx->dynamic_threads = config->dynamic_threads;
    
    // Initialize mutex and condition
    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        return ONEBIT_ERROR_THREAD;
    }
    
    if (pthread_cond_init(&ctx->cond, NULL) != 0) {
        pthread_mutex_destroy(&ctx->mutex);
        return ONEBIT_ERROR_THREAD;
    }
    
    // Allocate thread array
    ctx->threads = malloc(config->num_threads * sizeof(pthread_t));
    if (!ctx->threads) {
        pthread_mutex_destroy(&ctx->mutex);
        pthread_cond_destroy(&ctx->cond);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Create worker threads
    for (size_t i = 0; i < config->num_threads; i++) {
        if (pthread_create(&ctx->threads[i], NULL, thread_worker, ctx) != 0) {
            // Clean up on error
            ctx->should_stop = true;
            pthread_cond_broadcast(&ctx->cond);
            
            for (size_t j = 0; j < i; j++) {
                pthread_join(ctx->threads[j], NULL);
            }
            
            free(ctx->threads);
            pthread_mutex_destroy(&ctx->mutex);
            pthread_cond_destroy(&ctx->cond);
            return ONEBIT_ERROR_THREAD;
        }
    }
    
    return ONEBIT_SUCCESS;
}

void thread_pool_cleanup(ThreadPoolContext* ctx) {
    if (!ctx) return;
    
    pthread_mutex_lock(&ctx->mutex);
    ctx->should_stop = true;
    pthread_cond_broadcast(&ctx->cond);
    pthread_mutex_unlock(&ctx->mutex);
    
    // Wait for all threads to finish
    for (size_t i = 0; i < ctx->num_threads; i++) {
        pthread_join(ctx->threads[i], NULL);
    }
    
    // Clean up remaining tasks
    ThreadTask* task = ctx->task_queue;
    while (task) {
        ThreadTask* next = task->next;
        free(task);
        task = next;
    }
    
    free(ctx->threads);
    pthread_mutex_destroy(&ctx->mutex);
    pthread_cond_destroy(&ctx->cond);
}

int thread_pool_add_task(ThreadPoolContext* ctx, ThreadTaskFunc func, void* arg) {
    if (!ctx || !func) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ThreadTask* task = malloc(sizeof(ThreadTask));
    if (!task) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    task->func = func;
    task->arg = arg;
    task->next = NULL;
    
    pthread_mutex_lock(&ctx->mutex);
    
    // Check queue size limit
    if (ctx->max_queue_size > 0 && ctx->queue_size >= ctx->max_queue_size) {
        pthread_mutex_unlock(&ctx->mutex);
        free(task);
        return ONEBIT_ERROR_QUEUE_FULL;
    }
    
    // Add task to queue
    if (ctx->task_tail) {
        ctx->task_tail->next = task;
    } else {
        ctx->task_queue = task;
    }
    ctx->task_tail = task;
    ctx->queue_size++;
    
    pthread_cond_signal(&ctx->cond);
    pthread_mutex_unlock(&ctx->mutex);
    
    return ONEBIT_SUCCESS;
} 