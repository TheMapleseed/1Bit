#include "../test_utils.h"
#include "onebit_compute.h"
#include <time.h>

// Benchmark configuration
typedef struct {
    size_t matrix_size;
    size_t batch_size;
    size_t seq_length;
    size_t num_heads;
    size_t head_dim;
    int num_iterations;
} BenchConfig;

// Timing utilities
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// Matrix multiplication benchmark
void bench_matmul(ComputeContext* ctx, BenchConfig* config) {
    printf("\nMatrix Multiplication Benchmark\n");
    printf("==============================\n");
    
    size_t size = config->matrix_size;
    float* A = aligned_alloc(64, size * size * sizeof(float));
    float* B = aligned_alloc(64, size * size * sizeof(float));
    float* C = aligned_alloc(64, size * size * sizeof(float));
    
    // Initialize matrices
    for (size_t i = 0; i < size * size; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }
    
    // Warmup
    Matrix mat_a = {A, size, size, size, MATRIX_LAYOUT_ROW_MAJOR};
    Matrix mat_b = {B, size, size, size, MATRIX_LAYOUT_ROW_MAJOR};
    Matrix mat_c = {C, size, size, size, MATRIX_LAYOUT_ROW_MAJOR};
    
    struct { Matrix a, b, c; } params = {mat_a, mat_b, mat_c};
    compute_execute(ctx, "matrix_multiply", &params);
    
    // Benchmark
    double total_time = 0.0;
    for (int i = 0; i < config->num_iterations; i++) {
        double start = get_time_ms();
        compute_execute(ctx, "matrix_multiply", &params);
        double end = get_time_ms();
        total_time += (end - start);
    }
    
    double avg_time = total_time / config->num_iterations;
    double gflops = (2.0 * size * size * size) / (avg_time * 1e6);
    
    printf("Matrix size: %zux%zu\n", size, size);
    printf("Average time: %.2f ms\n", avg_time);
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    free(A);
    free(B);
    free(C);
}

// Attention mechanism benchmark
void bench_attention(ComputeContext* ctx, BenchConfig* config) {
    printf("\nAttention Mechanism Benchmark\n");
    printf("============================\n");
    
    size_t qkv_size = config->batch_size * config->num_heads * 
                      config->seq_length * config->head_dim;
    
    float* Q = aligned_alloc(64, qkv_size * sizeof(float));
    float* K = aligned_alloc(64, qkv_size * sizeof(float));
    float* V = aligned_alloc(64, qkv_size * sizeof(float));
    float* O = aligned_alloc(64, qkv_size * sizeof(float));
    
    // Initialize data
    for (size_t i = 0; i < qkv_size; i++) {
        Q[i] = (float)rand() / RAND_MAX;
        K[i] = (float)rand() / RAND_MAX;
        V[i] = (float)rand() / RAND_MAX;
    }
    
    // Create attention parameters
    AttentionConfig attn_config = {
        .batch_size = config->batch_size,
        .num_heads = config->num_heads,
        .seq_length = config->seq_length,
        .head_dim = config->head_dim,
        .attention_scale = 1.0f / sqrt(config->head_dim)
    };
    
    struct {
        float* query;
        float* key;
        float* value;
        float* output;
        AttentionConfig config;
    } params = {Q, K, V, O, attn_config};
    
    // Warmup
    compute_execute(ctx, "attention_forward", &params);
    
    // Benchmark
    double total_time = 0.0;
    for (int i = 0; i < config->num_iterations; i++) {
        double start = get_time_ms();
        compute_execute(ctx, "attention_forward", &params);
        double end = get_time_ms();
        total_time += (end - start);
    }
    
    double avg_time = total_time / config->num_iterations;
    double tflops = (2.0 * config->batch_size * config->num_heads * 
                    config->seq_length * config->seq_length * 
                    config->head_dim) / (avg_time * 1e9);
    
    printf("Batch size: %zu\n", config->batch_size);
    printf("Sequence length: %zu\n", config->seq_length);
    printf("Number of heads: %zu\n", config->num_heads);
    printf("Head dimension: %zu\n", config->head_dim);
    printf("Average time: %.2f ms\n", avg_time);
    printf("Performance: %.2f TFLOPS\n", tflops);
    
    free(Q);
    free(K);
    free(V);
    free(O);
} 