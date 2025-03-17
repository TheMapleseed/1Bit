#include "onebit.h"
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <time.h>

#define NUM_THREADS 8
#define TEST_DURATION 300  // 5 minutes
#define MAX_PROMPT_LENGTH 1024
#define MAX_OUTPUT_LENGTH 4096

typedef struct {
    int thread_id;
    OneBitContext* ctx;
    int num_iterations;
    int success_count;
    int error_count;
} ThreadData;

static void* stress_test_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    char prompt[MAX_PROMPT_LENGTH];
    char output[MAX_OUTPUT_LENGTH];
    
    for (int i = 0; i < data->num_iterations; i++) {
        snprintf(prompt, sizeof(prompt), "Thread %d iteration %d",
                data->thread_id, i);
        
        int result = onebit_generate(data->ctx, prompt, output,
                                  sizeof(output), 0.7f, 0.9f);
        
        if (result == ONEBIT_SUCCESS) {
            data->success_count++;
        } else {
            data->error_count++;
            printf("Thread %d error: %s\n", data->thread_id,
                   onebit_last_error());
        }
    }
    
    return NULL;
}

void test_concurrent_generation(void) {
    OneBitConfig config = onebit_default_config();
    config.vocab_file = "test_vocab.txt";
    config.num_threads = NUM_THREADS;
    
    OneBitContext ctx;
    assert(onebit_init(&ctx, &config) == ONEBIT_SUCCESS);
    
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];
    
    // Start threads
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].ctx = &ctx;
        thread_data[i].num_iterations = 1000;
        thread_data[i].success_count = 0;
        thread_data[i].error_count = 0;
        
        pthread_create(&threads[i], NULL, stress_test_thread, &thread_data[i]);
    }
    
    // Wait for threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Report results
    int total_iterations = 0;
    int total_success = 0;
    int total_errors = 0;
    
    for (int i = 0; i < NUM_THREADS; i++) {
        total_iterations += thread_data[i].num_iterations;
        total_success += thread_data[i].success_count;
        total_errors += thread_data[i].error_count;
    }
    
    printf("Stress test results:\n");
    printf("Total iterations: %d\n", total_iterations);
    printf("Successful generations: %d\n", total_success);
    printf("Failed generations: %d\n", total_errors);
    printf("Success rate: %.2f%%\n",
           (float)total_success / total_iterations * 100);
}

// Memory stress test
void test_memory_stress(void) {
    OneBitConfig config = onebit_default_config();
    config.vocab_file = "test_vocab.txt";
    config.num_threads = NUM_THREADS;
    config.memory_pool_size = 512 * 1024 * 1024;  // 512MB
    
    OneBitContext ctx;
    assert(onebit_init(&ctx, &config) == ONEBIT_SUCCESS);
    
    char output[MAX_OUTPUT_LENGTH];
    const char* long_prompt = "This is a very long prompt that will be repeated multiple times to stress test the memory system. ";
    char* extended_prompt = malloc(MAX_PROMPT_LENGTH);
    
    // Create a very long prompt by repeating the base prompt
    int base_len = strlen(long_prompt);
    for (int i = 0; i < MAX_PROMPT_LENGTH/base_len; i++) {
        strncpy(extended_prompt + i*base_len, long_prompt, base_len);
    }
    extended_prompt[MAX_PROMPT_LENGTH-1] = '\0';
    
    // Repeatedly generate with long prompts
    for (int i = 0; i < 100; i++) {
        assert(onebit_generate(&ctx, extended_prompt, output,
                             sizeof(output), 0.7f, 0.9f) == ONEBIT_SUCCESS);
    }
    
    free(extended_prompt);
    onebit_cleanup(&ctx);
    printf("Memory stress test passed\n");
}

// Resource exhaustion test
void test_resource_exhaustion(void) {
    OneBitConfig config = onebit_default_config();
    config.vocab_file = "test_vocab.txt";
    config.num_threads = NUM_THREADS;
    
    // Test with minimal memory pool
    config.memory_pool_size = 1024 * 1024;  // 1MB (intentionally small)
    
    OneBitContext ctx;
    int result = onebit_init(&ctx, &config);
    
    // Should fail gracefully
    assert(result != ONEBIT_SUCCESS);
    assert(strstr(onebit_last_error(), "memory") != NULL);
    
    // Test with excessive threads
    config.memory_pool_size = 1024 * 1024 * 1024;  // 1GB
    config.num_threads = 1000000;  // Excessive number
    
    result = onebit_init(&ctx, &config);
} 