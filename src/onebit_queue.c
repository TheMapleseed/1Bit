#include "onebit/onebit_queue.h"
#include "onebit/onebit_error.h"
#include <string.h>

// Queue node
typedef struct QueueNode {
    void* data;
    size_t size;
    struct QueueNode* next;
} QueueNode;

int queue_init(QueueContext* ctx, const QueueConfig* config) {
    if (!ctx) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->head = NULL;
    ctx->tail = NULL;
    ctx->size = 0;
    ctx->max_size = config ? config->max_size : 0;
    
    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        return ONEBIT_ERROR_THREAD;
    }
    
    if (pthread_cond_init(&ctx->not_empty, NULL) != 0) {
        pthread_mutex_destroy(&ctx->mutex);
        return ONEBIT_ERROR_THREAD;
    }
    
    if (pthread_cond_init(&ctx->not_full, NULL) != 0) {
        pthread_cond_destroy(&ctx->not_empty);
        pthread_mutex_destroy(&ctx->mutex);
        return ONEBIT_ERROR_THREAD;
    }
    
    return ONEBIT_SUCCESS;
}

void queue_cleanup(QueueContext* ctx) {
    if (!ctx) return;
    
    pthread_mutex_lock(&ctx->mutex);
    
    // Free all nodes
    QueueNode* node = ctx->head;
    while (node) {
        QueueNode* next = node->next;
        free(node->data);
        free(node);
        node = next;
    }
    
    pthread_mutex_unlock(&ctx->mutex);
    
    pthread_cond_destroy(&ctx->not_empty);
    pthread_cond_destroy(&ctx->not_full);
    pthread_mutex_destroy(&ctx->mutex);
}

int queue_push(QueueContext* ctx, const void* data,
              size_t size, int timeout_ms) {
    if (!ctx || !data || size == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    // Wait if queue is full
    if (ctx->max_size > 0) {
        struct timespec ts;
        if (timeout_ms > 0) {
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += timeout_ms / 1000;
            ts.tv_nsec += (timeout_ms % 1000) * 1000000;
            if (ts.tv_nsec >= 1000000000) {
                ts.tv_sec++;
                ts.tv_nsec -= 1000000000;
            }
        }
        
        while (ctx->size >= ctx->max_size) {
            if (timeout_ms < 0) {
                pthread_cond_wait(&ctx->not_full, &ctx->mutex);
            } else if (timeout_ms == 0) {
                pthread_mutex_unlock(&ctx->mutex);
                return ONEBIT_ERROR_TIMEOUT;
            } else {
                int result = pthread_cond_timedwait(&ctx->not_full,
                                                  &ctx->mutex, &ts);
                if (result == ETIMEDOUT) {
                    pthread_mutex_unlock(&ctx->mutex);
                    return ONEBIT_ERROR_TIMEOUT;
                }
            }
        }
    }
    
    // Create new node
    QueueNode* node = malloc(sizeof(QueueNode));
    if (!node) {
        pthread_mutex_unlock(&ctx->mutex);
        return ONEBIT_ERROR_MEMORY;
    }
    
    node->data = malloc(size);
    if (!node->data) {
        free(node);
        pthread_mutex_unlock(&ctx->mutex);
        return ONEBIT_ERROR_MEMORY;
    }
    
    memcpy(node->data, data, size);
    node->size = size;
    node->next = NULL;
    
    // Add to queue
    if (ctx->tail) {
        ctx->tail->next = node;
    } else {
        ctx->head = node;
    }
    ctx->tail = node;
    ctx->size++;
    
    pthread_cond_signal(&ctx->not_empty);
    pthread_mutex_unlock(&ctx->mutex);
    
    return ONEBIT_SUCCESS;
}

int queue_pop(QueueContext* ctx, void* data, size_t* size,
             int timeout_ms) {
    if (!ctx || !data || !size) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    // Wait if queue is empty
    struct timespec ts;
    if (timeout_ms > 0) {
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += timeout_ms / 1000;
        ts.tv_nsec += (timeout_ms % 1000) * 1000000;
        if (ts.tv_nsec >= 1000000000) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000;
        }
    }
    
    while (!ctx->head) {
        if (timeout_ms < 0) {
            pthread_cond_wait(&ctx->not_empty, &ctx->mutex);
        } else if (timeout_ms == 0) {
            pthread_mutex_unlock(&ctx->mutex);
            return ONEBIT_ERROR_TIMEOUT;
        } else {
            int result = pthread_cond_timedwait(&ctx->not_empty,
                                              &ctx->mutex, &ts);
            if (result == ETIMEDOUT) {
                pthread_mutex_unlock(&ctx->mutex);
                return ONEBIT_ERROR_TIMEOUT;
            }
        }
    }
    
    // Remove node
    QueueNode* node = ctx->head;
    ctx->head = node->next;
    if (!ctx->head) {
        ctx->tail = NULL;
    }
    ctx->size--;
    
    memcpy(data, node->data, node->size);
    *size = node->size;
    
    free(node->data);
    free(node);
    
    pthread_cond_signal(&ctx->not_full);
    pthread_mutex_unlock(&ctx->mutex);
    
    return ONEBIT_SUCCESS;
} 