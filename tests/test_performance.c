#include "onebit.h"
#include <stdio.h>
#include <assert.h>
#include <time.h>

double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

void test_generation_speed(void) {
    OneBitConfig config = onebit_default_config();
    config.vocab_file = "test_vocab.txt";
    
    OneBitContext ctx;
    assert(onebit_init(&ctx, &config) == ONEBIT_SUCCESS);
    
    const char* prompt = "Testing generation speed";
    char output[4096];
    int num_runs = 100;
    
    double start_time = get_time_ms();
    for (int i = 0; i < num_runs; i++) {
        assert(onebit_generate(&ctx, prompt, output, sizeof(output),
                            0.7f, 0.9f) == ONEBIT_SUCCESS);
    }
    double end_time = get_time_ms();
    
    double avg_time = (end_time - start_time) / num_runs;
    printf("Average generation time: %.2f ms\n", avg_time);
    
    onebit_cleanup(&ctx);
}

void test_memory_usage(void) {
    OneBitConfig config = onebit_default_config();
    config.vocab_file = "test_vocab.txt";
    
    OneBitContext ctx;
    assert(onebit_init(&ctx, &config) == ONEBIT_SUCCESS);
    
    size_t initial_memory = ctx.pool->used_size;
    printf("Initial memory usage: %zu bytes\n", initial_memory);
    
    // Run some operations
    char output[4096];
    for (int i = 0; i < 10; i++) {
        assert(onebit_generate(&ctx, "Memory test", output,
                            sizeof(output), 0.7f, 0.9f) == ONEBIT_SUCCESS);
    }
    
    size_t final_memory = ctx.pool->used_size;
    printf("Final memory usage: %zu bytes\n", final_memory);
    printf("Memory growth: %zu bytes\n", final_memory - initial_memory);
    
    onebit_cleanup(&ctx);
}

void test_cuda_performance(void) {
    OneBitConfig config = onebit_default_config();
    config.vocab_file = "test_vocab.txt";
    config.use_cuda = true;
    
    OneBitContext ctx;
    if (onebit_init(&ctx, &config) != ONEBIT_SUCCESS) {
        printf("CUDA not available, skipping test\n");
        return;
    }
    
    // Test CUDA operations
    char output[4096];
    double start_time = get_time_ms();
    
    for (int i = 0; i < 100; i++) {
        assert(onebit_generate(&ctx, "CUDA test", output,
                            sizeof(output), 0.7f, 0.9f) == ONEBIT_SUCCESS);
    }
    
    double end_time = get_time_ms();
    printf("CUDA average time: %.2f ms\n", (end_time - start_time) / 100);
    
    onebit_cleanup(&ctx);
}

int main(void) {
    printf("Running performance tests...\n");
    
    test_generation_speed();
    test_memory_usage();
    test_cuda_performance();
    
    printf("All performance tests completed!\n");
    return 0;
} 