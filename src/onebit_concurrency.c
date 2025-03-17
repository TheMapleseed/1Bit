#include "onebit/onebit_concurrency.h"
#include "onebit/onebit_error.h"
#include <string.h>

// Thread pool implementation
static void* worker_thread(void* arg) {
    ThreadPool* pool = (ThreadPool*)arg;
    
    while (true) {
        pthread_mutex_lock(&pool->mutex);
        
        while (pool->num_tasks == 0 && !pool->shutdown) {
            pthread_cond_wait(&pool->condition, &pool->mutex);
        }
        
        if (pool->shutdown && pool->num_tasks == 0) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }
        
        // Get highest priority task
        Task* task = NULL;
        int highest_priority = -1;
        
        for (int i = 0; i < pool->num_tasks; i++) {
            if (pool->tasks[i].priority > highest_priority) {
                highest_priority = pool->tasks[i].priority;
                task = &pool->tasks[i];
            }
        }
        
        if (task) {
            // Remove task from queue
            memmove(task, task + 1,
                   (pool->num_tasks - 1) * sizeof(Task));
            pool->num_tasks--;
            
            pthread_mutex_unlock(&pool->mutex);
            
            // Execute task
            task->function(task->arg);
        } else {
            pthread_mutex_unlock(&pool->mutex);
        }
    }
    
    return NULL;
}

int thread_pool_init(ThreadPool* pool, int num_threads) {
    if (!pool || num_threads <= 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pool->threads = malloc(num_threads * sizeof(pthread_t));
    if (!pool->threads) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    pool->tasks = malloc(MAX_TASKS * sizeof(Task));
    if (!pool->tasks) {
        free(pool->threads);
        return ONEBIT_ERROR_MEMORY;
    }
    
    pool->num_threads = num_threads;
    pool->num_tasks = 0;
    pool->shutdown = false;
    
    if (pthread_mutex_init(&pool->mutex, NULL) != 0) {
        free(pool->threads);
        free(pool->tasks);
        return ONEBIT_ERROR_THREAD;
    }
    
    if (pthread_cond_init(&pool->condition, NULL) != 0) {
        pthread_mutex_destroy(&pool->mutex);
        free(pool->threads);
        free(pool->tasks);
        return ONEBIT_ERROR_THREAD;
    }
    
    // Create worker threads
    for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&pool->threads[i], NULL,
                          worker_thread, pool) != 0) {
            thread_pool_cleanup(pool);
            return ONEBIT_ERROR_THREAD;
        }
    }
    
    return ONEBIT_SUCCESS;
}

void thread_pool_cleanup(ThreadPool* pool) {
    if (!pool) return;
    
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = true;
    pthread_cond_broadcast(&pool->condition);
    pthread_mutex_unlock(&pool->mutex);
    
    for (int i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->condition);
    free(pool->threads);
    free(pool->tasks);
}

int thread_pool_submit(ThreadPool* pool, TaskFunction function,
                      void* arg, int priority) {
    if (!pool || !function) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&pool->mutex);
    
    if (pool->num_tasks >= MAX_TASKS) {
        pthread_mutex_unlock(&pool->mutex);
        return ONEBIT_ERROR_OVERFLOW;
    }
    
    Task* task = &pool->tasks[pool->num_tasks++];
    task->function = function;
    task->arg = arg;
    task->priority = priority;
    
    pthread_cond_signal(&pool->condition);
    pthread_mutex_unlock(&pool->mutex);
    
    return ONEBIT_SUCCESS;
} 