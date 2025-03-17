#include "onebit/onebit_async.h"
#include "onebit/onebit_concurrency.h"
#include "onebit/onebit_error.h"
#include <stdlib.h>
#include <string.h>

// Internal task wrapper for promise resolution
typedef struct {
    AsyncTask task;
    Promise* promise;
    void* output;
    size_t output_size;
    int error_code;
    char error_message[256];
} TaskWrapper;

static void task_cleanup(TaskWrapper* wrapper) {
    if (wrapper->output) {
        free(wrapper->output);
    }
    free(wrapper);
}

static void task_complete(void* arg) {
    TaskWrapper* wrapper = (TaskWrapper*)arg;
    Promise* promise = wrapper->promise;
    
    if (wrapper->error_code) {
        promise_reject(promise, wrapper->error_code, wrapper->error_message);
    } else {
        promise_resolve(promise, wrapper->output);
    }
    
    task_cleanup(wrapper);
}

Promise* async_run(AsyncTask* task, int priority) {
    TaskWrapper* wrapper = malloc(sizeof(TaskWrapper));
    if (!wrapper) {
        return NULL;
    }
    
    memcpy(&wrapper->task, task, sizeof(AsyncTask));
    wrapper->promise = promise_create();
    wrapper->output = NULL;
    wrapper->output_size = 0;
    wrapper->error_code = 0;
    memset(wrapper->error_message, 0, sizeof(wrapper->error_message));
    
    if (!wrapper->promise) {
        free(wrapper);
        return NULL;
    }
    
    int result = thread_pool_submit(get_global_pool(),
                                  task_complete, wrapper, priority);
    
    if (result != ONEBIT_SUCCESS) {
        promise_reject(wrapper->promise, result, onebit_last_error());
        task_cleanup(wrapper);
    }
    
    return wrapper->promise;
}

void* await_promise(Promise* promise) {
    if (!promise) {
        throw(ONEBIT_ERROR_INVALID_PARAM);
    }
    
    TaskWrapper* task = (TaskWrapper*)promise_get_context(promise);
    if (!task) {
        throw(ONEBIT_ERROR_INVALID_PARAM);
    }
    
    return task->output;
} 