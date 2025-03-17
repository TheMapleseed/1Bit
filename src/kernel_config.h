#ifndef KERNEL_CONFIG_H
#define KERNEL_CONFIG_H

#include <stdint.h>

// Maximum number of supported kernel configurations
#define MAX_KERNEL_CONFIGS 16

// Kernel shape configuration
typedef struct {
    int M;          // Matrix rows
    int K;          // Matrix columns
    int BM;         // Block size for M dimension
    int BK;         // Block size for K dimension
    int bmm;        // Micro-block size
} KernelShape;

// Kernel configuration container
typedef struct {
    KernelShape shapes[MAX_KERNEL_CONFIGS];
    int num_configs;
    int initialized;
} KernelConfigContainer;

// Function declarations
void init_kernel_container(void);
int add_kernel_config(int M, int K, int BM, int BK, int bmm);
const KernelShape* get_kernel_config(int M, int K);
void cleanup_kernel_configs(void);

// Platform-specific kernel selection
#ifdef __ARM_NEON
#define USE_TL1_KERNELS
#elif defined(__AVX2__)
#define USE_TL2_KERNELS
#endif

#endif 