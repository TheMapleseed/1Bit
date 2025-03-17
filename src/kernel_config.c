#include "kernel_config.h"
#include <stdio.h>
#include <stdlib.h>

// Global kernel configuration container
static KernelConfigContainer g_kernel_container = {0};

void init_kernel_container(void) {
    if (g_kernel_container.initialized) {
        return;
    }

    g_kernel_container.num_configs = 0;
    g_kernel_container.initialized = 1;

    // Initialize default configurations based on platform
    #ifdef USE_TL1_KERNELS
    // TL1 (ARM) configurations
    add_kernel_config(3200, 8640, 160, 64, 32);  // 3B model config
    add_kernel_config(3200, 3200, 320, 128, 64);
    add_kernel_config(8640, 3200, 320, 64, 32);
    
    add_kernel_config(1536, 4096, 256, 128, 32); // Large model config
    add_kernel_config(1536, 1536, 128, 64, 64);
    add_kernel_config(4096, 1536, 256, 128, 32);

    #elif defined(USE_TL2_KERNELS)
    // TL2 (x86) configurations
    add_kernel_config(3200, 8640, 160, 96, 32);  // 3B model config
    add_kernel_config(3200, 3200, 320, 96, 32);
    add_kernel_config(8640, 3200, 320, 96, 32);
    
    add_kernel_config(1536, 4096, 256, 96, 32);  // Large model config
    add_kernel_config(1536, 1536, 128, 192, 32);
    add_kernel_config(4096, 1536, 256, 96, 32);

    #else
    // Generic configurations
    add_kernel_config(3200, 8640, 128, 32, 16);
    add_kernel_config(1536, 4096, 128, 32, 16);
    #endif
}

int add_kernel_config(int M, int K, int BM, int BK, int bmm) {
    if (!g_kernel_container.initialized || 
        g_kernel_container.num_configs >= MAX_KERNEL_CONFIGS) {
        return -1;
    }

    KernelShape* shape = &g_kernel_container.shapes[g_kernel_container.num_configs];
    shape->M = M;
    shape->K = K;
    shape->BM = BM;
    shape->BK = BK;
    shape->bmm = bmm;

    g_kernel_container.num_configs++;
    return 0;
}

const KernelShape* get_kernel_config(int M, int K) {
    if (!g_kernel_container.initialized) {
        return NULL;
    }

    // Find matching configuration
    for (int i = 0; i < g_kernel_container.num_configs; i++) {
        const KernelShape* shape = &g_kernel_container.shapes[i];
        if (shape->M == M && shape->K == K) {
            return shape;
        }
    }

    // Find best matching configuration
    const KernelShape* best_match = NULL;
    int min_diff = INT_MAX;

    for (int i = 0; i < g_kernel_container.num_configs; i++) {
        const KernelShape* shape = &g_kernel_container.shapes[i];
        int diff = abs(shape->M - M) + abs(shape->K - K);
        
        if (diff < min_diff) {
            min_diff = diff;
            best_match = shape;
        }
    }

    return best_match;
}

void cleanup_kernel_configs(void) {
    g_kernel_container.num_configs = 0;
    g_kernel_container.initialized = 0;
}

// Helper function to validate kernel parameters
int validate_kernel_params(const KernelShape* shape) {
    if (!shape) return -1;

    // Validate divisibility
    if (shape->M % shape->BM != 0) return -1;
    if (shape->K % shape->BK != 0) return -1;
    if (shape->BM % shape->bmm != 0) return -1;

    // Validate minimum sizes
    if (shape->M < 128) return -1;
    if (shape->K < 128) return -1;
    if (shape->BM < 16) return -1;
    if (shape->BK < 16) return -1;
    if (shape->bmm < 8) return -1;

    return 0;
} 