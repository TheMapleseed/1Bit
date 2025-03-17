/**
 * @file onebit_compute_opt.c
 * @brief Implementation of compute optimizations
 */

#include "onebit/onebit_compute_opt.h"
#include "onebit/onebit_error.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#ifdef __x86_64__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// Global optimization context
static struct {
    OptConfig config;
    pthread_t* threads;
    bool initialized;
} g_opt_ctx = {0};

// Thread pool task
typedef struct {
    void (*func)(void*);
    void* arg;
} Task;

// CPU feature detection
#ifdef __x86_64__
static void detect_x86_features(CPUFeatures* features) {
    int cpu_info[4];
    
    // Get CPU features using CPUID
    __cpuid(1, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
    
    features->has_sse = (cpu_info[3] & (1 << 25)) != 0;
    features->has_sse2 = (cpu_info[3] & (1 << 26)) != 0;
    features->has_sse3 = (cpu_info[2] & (1 << 0)) != 0;
    features->has_ssse3 = (cpu_info[2] & (1 << 9)) != 0;
    features->has_sse4_1 = (cpu_info[2] & (1 << 19)) != 0;
    features->has_sse4_2 = (cpu_info[2] & (1 << 20)) != 0;
    features->has_avx = (cpu_info[2] & (1 << 28)) != 0;
    
    // Check for AVX2 and FMA
    if (__get_cpuid_max(0, NULL) >= 7) {
        __cpuid_count(7, 0, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
        features->has_avx2 = (cpu_info[1] & (1 << 5)) != 0;
        features->has_avx512f = (cpu_info[1] & (1 << 16)) != 0;
    }
    
    features->has_fma = (cpu_info[2] & (1 << 12)) != 0;
    features->has_neon = false;
}
#endif

#ifdef __ARM_NEON
static void detect_arm_features(CPUFeatures* features) {
    memset(features, 0, sizeof(CPUFeatures));
    features->has_neon = true;
}
#endif

// Optimized matrix multiplication using SIMD
static void matmul_simd(const float* A, const float* B, float* C,
                       size_t M, size_t N, size_t K) {
#ifdef __x86_64__
    if (g_opt_ctx.config.use_simd && g_opt_ctx.config.use_cache_opt) {
        // Block size based on L1 cache size
        const size_t block_size = sqrt(g_opt_ctx.config.l1_cache_size / sizeof(float) / 3);
        
        for (size_t i = 0; i < M; i += block_size) {
            for (size_t j = 0; j < N; j += block_size) {
                for (size_t k = 0; k < K; k += block_size) {
                    size_t i_end = (i + block_size < M) ? i + block_size : M;
                    size_t j_end = (j + block_size < N) ? j + block_size : N;
                    size_t k_end = (k + block_size < K) ? k + block_size : K;
                    
                    for (size_t ii = i; ii < i_end; ii += 8) {
                        for (size_t jj = j; jj < j_end; jj += 8) {
                            __m256 sum[8] = {_mm256_setzero_ps()};
                            
                            for (size_t kk = k; kk < k_end; kk++) {
                                __m256 a = _mm256_load_ps(&A[ii * K + kk]);
                                __m256 b = _mm256_broadcast_ss(&B[kk * N + jj]);
                                sum[0] = _mm256_fmadd_ps(a, b, sum[0]);
                            }
                            
                            _mm256_store_ps(&C[ii * N + jj], sum[0]);
                        }
                    }
                }
            }
        }
    }
#endif

#ifdef __ARM_NEON
    if (g_opt_ctx.config.use_simd) {
        // ARM NEON implementation
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j += 4) {
                float32x4_t sum = vdupq_n_f32(0);
                
                for (size_t k = 0; k < K; k++) {
                    float32x4_t a = vld1q_dup_f32(&A[i * K + k]);
                    float32x4_t b = vld1q_f32(&B[k * N + j]);
                    sum = vmlaq_f32(sum, a, b);
                }
                
                vst1q_f32(&C[i * N + j], sum);
            }
        }
    }
#endif
}

// Thread function for parallel matrix multiplication
typedef struct {
    const float* A;
    const float* B;
    float* C;
    size_t M, N, K;
    size_t start_row;
    size_t end_row;
// CPU SIMD implementations
static inline void binary_gemm_avx512(const uint8_t* A,
                                    const uint8_t* B,
                                    float* C,
                                    size_t M,
                                    size_t N,
                                    size_t K,
                                    float scale) {
    #ifdef __AVX512F__
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            __m512i sum = _mm512_setzero_si512();
            
            for (size_t k = 0; k < K; k += 512) {
                __m512i a = _mm512_load_si512((__m512i*)&A[m * K + k]);
                __m512i b = _mm512_load_si512((__m512i*)&B[n * K + k]);
                
                // XNOR operation
                __m512i xnor = _mm512_xor_si512(a, b);
                xnor = _mm512_xor_si512(xnor, _mm512_set1_epi8(-1));
                
                // Population count
                sum = _mm512_add_epi32(sum, _mm512_popcnt_epi32(xnor));
            }
            
            // Horizontal sum and scaling
            int total = _mm512_reduce_add_epi32(sum);
            C[m * N + n] = total * scale;
        }
    }
    #endif
}

static inline void quantize_avx512(const float* input,
                                 uint8_t* output,
                                 size_t size,
                                 float scale) {
    #ifdef __AVX512F__
    __m512 scale_vec = _mm512_set1_ps(scale);
    
    for (size_t i = 0; i < size; i += 16) {
        __m512 in = _mm512_load_ps(&input[i]);
        __m512 scaled = _mm512_mul_ps(in, scale_vec);
        __m512i rounded = _mm512_cvtps_epi32(scaled);
        
        // Pack to 8-bit with saturation
        __m256i packed = _mm512_cvtepi32_epi8(rounded);
        _mm256_storeu_si256((__m256i*)&output[i], packed);
    }
    #endif
}

// GPU implementations
#ifdef USE_CUDA
static int binary_gemm_cuda(ComputeContext* ctx,
                           const MatrixDescriptor* A,
                           const MatrixDescriptor* B,
                           MatrixDescriptor* C,
                           float scale) {
    // Custom CUDA kernel for binary GEMM
    // Implementation details omitted for brevity
    return ONEBIT_SUCCESS;
}
#endif

int compute_init(ComputeContext* ctx) {
    if (!ctx) return ONEBIT_ERROR_INVALID;
    
    memset(ctx, 0, sizeof(ComputeContext));
    
    // Detect CPU capabilities
    #if defined(__x86_64__)
        int cpu_info[4];
        __cpuid(1, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
        
        ctx->hw_caps.has_sse4 = (cpu_info[2] & (1 << 19)) != 0;
        ctx->hw_caps.has_avx2 = (cpu_info[2] & (1 << 28)) != 0;
        
        __cpuid_count(7, 0, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
        ctx->hw_caps.has_avx512 = (cpu_info[1] & (1 << 16)) != 0;
    #elif defined(__aarch64__)
        ctx->hw_caps.has_neon = true;
    #endif
    
    // Initialize GPU if available
    #ifdef USE_CUDA
    cudaError_t cuda_err = cudaGetDeviceCount(&ctx->hw_caps.num_gpu_devices);
    if (cuda_err == cudaSuccess && ctx->hw_caps.num_gpu_devices > 0) {
        ctx->hw_caps.cuda_available = true;
        
        for (int i = 0; i < ctx->hw_caps.num_gpu_devices; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            ctx->hw_caps.gpu_memory[i] = prop.totalGlobalMem;
            
            // Create CUDA streams
            cudaStreamCreate(&ctx->cuda_streams[i]);
        }
    }
    #endif
    
    // Allocate workspace
    ctx->workspace_size = 1024 * 1024 * 1024;  // 1GB default
    ctx->workspace = aligned_alloc(4096, ctx->workspace_size);
    if (!ctx->workspace) return ONEBIT_ERROR_MEMORY;
    
    return ONEBIT_SUCCESS;
}

int compute_binary_gemm(ComputeContext* ctx,
                       const MatrixDescriptor* A,
                       const MatrixDescriptor* B,
                       MatrixDescriptor* C,
                       float scale) {
    if (!ctx || !A || !B || !C) return ONEBIT_ERROR_INVALID;
    
    // Choose optimal implementation based on hardware
    if (A->on_device || B->on_device || C->on_device) {
        #ifdef USE_CUDA
        if (ctx->hw_caps.cuda_available) {
            return binary_gemm_cuda(ctx, A, B, C, scale);
        }
        #endif
        return ONEBIT_ERROR_INVALID;
    } else {
        // CPU implementation
        if (ctx->hw_caps.has_avx512) {
            binary_gemm_avx512((uint8_t*)A->data,
                             (uint8_t*)B->data,
                             (float*)C->data,
                             A->rows, B->cols, A->cols,
                             scale);
        } else {
            // Fallback implementation
            // ...
        }
    }
    
    return ONEBIT_SUCCESS;
} 