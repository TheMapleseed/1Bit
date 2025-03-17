#include "onebit/onebit_compute_loader.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

struct KernelLoaderContext {
    void* handle;
    ComputeDeviceType device_type;
    char last_error[256];
};

#ifdef __APPLE__
    #define LIB_EXTENSION ".dylib"
#else
    #define LIB_EXTENSION ".so"
#endif

int loader_init(KernelLoaderContext** ctx, ComputeDeviceType device_type) {
    *ctx = malloc(sizeof(KernelLoaderContext));
    if (!*ctx) return -1;
    
    (*ctx)->handle = NULL;
    (*ctx)->device_type = device_type;
    (*ctx)->last_error[0] = '\0';
    
    return 0;
}

int loader_reload_kernel(KernelLoaderContext* ctx, const char* kernel_path) {
    if (ctx->handle) {
        dlclose(ctx->handle);
    }
    
    ctx->handle = dlopen(kernel_path, RTLD_NOW);
    if (!ctx->handle) {
        strncpy(ctx->last_error, dlerror(), sizeof(ctx->last_error) - 1);
        return -1;
    }
    
    return 0;
}

void* loader_get_kernel(KernelLoaderContext* ctx, const char* kernel_name) {
    if (!ctx->handle) return NULL;
    return dlsym(ctx->handle, kernel_name);
}

void loader_cleanup(KernelLoaderContext* ctx) {
    if (ctx) {
        if (ctx->handle) {
            dlclose(ctx->handle);
        }
        free(ctx);
    }
} 