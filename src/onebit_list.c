#include "onebit/onebit_list.h"
#include "onebit/onebit_error.h"
#include <string.h>

// List node
typedef struct ListNode {
    void* data;
    size_t size;
    struct ListNode* prev;
    struct ListNode* next;
} ListNode;

int list_init(ListContext* ctx, const ListConfig* config) {
    if (!ctx) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->head = NULL;
    ctx->tail = NULL;
    ctx->size = 0;
    
    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        return ONEBIT_ERROR_THREAD;
    }
    
    return ONEBIT_SUCCESS;
}

void list_cleanup(ListContext* ctx) {
    if (!ctx) return;
    
    pthread_mutex_lock(&ctx->mutex);
    
    ListNode* node = ctx->head;
    while (node) {
        ListNode* next = node->next;
        free(node->data);
        free(node);
        node = next;
    }
    
    pthread_mutex_unlock(&ctx->mutex);
    pthread_mutex_destroy(&ctx->mutex);
}

int list_push_front(ListContext* ctx, const void* data,
                    size_t size) {
    if (!ctx || !data || size == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ListNode* node = malloc(sizeof(ListNode));
    if (!node) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    node->data = malloc(size);
    if (!node->data) {
        free(node);
        return ONEBIT_ERROR_MEMORY;
    }
    
    memcpy(node->data, data, size);
    node->size = size;
    
    pthread_mutex_lock(&ctx->mutex);
    
    node->prev = NULL;
    node->next = ctx->head;
    
    if (ctx->head) {
        ctx->head->prev = node;
    } else {
        ctx->tail = node;
    }
    
    ctx->head = node;
    ctx->size++;
    
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
}

int list_push_back(ListContext* ctx, const void* data,
                   size_t size) {
    if (!ctx || !data || size == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ListNode* node = malloc(sizeof(ListNode));
    if (!node) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    node->data = malloc(size);
    if (!node->data) {
        free(node);
        return ONEBIT_ERROR_MEMORY;
    }
    
    memcpy(node->data, data, size);
    node->size = size;
    
    pthread_mutex_lock(&ctx->mutex);
    
    node->next = NULL;
    node->prev = ctx->tail;
    
    if (ctx->tail) {
        ctx->tail->next = node;
    } else {
        ctx->head = node;
    }
    
    ctx->tail = node;
    ctx->size++;
    
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
}

int list_pop_front(ListContext* ctx, void* data, size_t* size) {
    if (!ctx || !data || !size || !ctx->head) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    ListNode* node = ctx->head;
    memcpy(data, node->data, node->size);
    *size = node->size;
    
    ctx->head = node->next;
    if (ctx->head) {
        ctx->head->prev = NULL;
    } else {
        ctx->tail = NULL;
    }
    
    ctx->size--;
    
    free(node->data);
    free(node);
    
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
}

int list_pop_back(ListContext* ctx, void* data, size_t* size) {
    if (!ctx || !data || !size || !ctx->tail) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    ListNode* node = ctx->tail;
    memcpy(data, node->data, node->size);
    *size = node->size;
    
    ctx->tail = node->prev;
    if (ctx->tail) {
        ctx->tail->next = NULL;
    } else {
        ctx->head = NULL;
    }
    
    ctx->size--;
    
    free(node->data);
    free(node);
    
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
}

int list_front(const ListContext* ctx, void* data,
               size_t* size) {
    if (!ctx || !data || !size || !ctx->head) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock((pthread_mutex_t*)&ctx->mutex);
    
    memcpy(data, ctx->head->data, ctx->head->size);
    *size = ctx->head->size;
    
    pthread_mutex_unlock((pthread_mutex_t*)&ctx->mutex);
    return ONEBIT_SUCCESS;
}

int list_back(const ListContext* ctx, void* data,
              size_t* size) {
    if (!ctx || !data || !size || !ctx->tail) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock((pthread_mutex_t*)&ctx->mutex);
    
    memcpy(data, ctx->tail->data, ctx->tail->size);
    *size = ctx->tail->size;
    
    pthread_mutex_unlock((pthread_mutex_t*)&ctx->mutex);
    return ONEBIT_SUCCESS;
}

void list_clear(ListContext* ctx) {
    if (!ctx) return;
} 