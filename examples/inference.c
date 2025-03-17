#include <onebit/onebit.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
 * @brief Example application for OneBit model inference
 * 
 * This example loads a model and performs text generation
 * using the OneBit library.
 */
int main(int argc, char** argv) {
    // Check arguments
    if (argc < 2) {
        printf("Usage: %s <model_path> [prompt]\n", argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    const char* prompt = (argc > 2) ? argv[2] : "Hello, my name is";
    
    // Initialize OneBit context
    OneBitContext ctx;
    OneBitConfig config = onebit_default_config();
    
    // Configure hardware options
    config.use_cuda = onebit_get_device_count() > 0;
    config.device_id = 0;
    config.num_threads = 8;
    config.use_memory_pool = true;
    config.memory_pool_size = 1024 * 1024 * 1024;  // 1GB
    
    // Initialize OneBit
    int result = onebit_init(&ctx, &config);
    if (result != ONEBIT_SUCCESS) {
        printf("Error initializing OneBit: %s\n", onebit_error_string(result));
        return 1;
    }
    
    printf("OneBit v%s initialized\n", onebit_version());
    
    // Print hardware capabilities
    printf("Hardware capabilities:\n");
    printf("  AVX-512: %s\n", onebit_has_avx512() ? "Yes" : "No");
    printf("  AVX2   : %s\n", onebit_has_avx2() ? "Yes" : "No");
    printf("  SSE4   : %s\n", onebit_has_sse4() ? "Yes" : "No");
    printf("  GPUs   : %d\n", onebit_get_device_count());
    
    // Load model
    printf("Loading model from %s...\n", model_path);
    result = onebit_load_model(&ctx, model_path);
    if (result != ONEBIT_SUCCESS) {
        printf("Error loading model: %s\n", onebit_error_string(result));
        onebit_cleanup(&ctx);
        return 1;
    }
    
    // Perform text generation
    printf("\nPrompt: %s\n", prompt);
    printf("Generating text...\n\n");
    
    const size_t output_size = 1024;
    char* output = malloc(output_size);
    if (!output) {
        printf("Memory allocation error\n");
        onebit_cleanup(&ctx);
        return 1;
    }
    
    // Start timing
    clock_t start = clock();
    
    // Generate text
    result = onebit_generate(&ctx, prompt, output, output_size, 0.8f, 0.9f);
    if (result != ONEBIT_SUCCESS) {
        printf("Error generating text: %s\n", onebit_error_string(result));
        free(output);
        onebit_cleanup(&ctx);
        return 1;
    }
    
    // End timing
    clock_t end = clock();
    float elapsed = (float)(end - start) / CLOCKS_PER_SEC;
    
    // Print result
    printf("%s\n\n", output);
    printf("Generation completed in %.2f seconds\n", elapsed);
    
    // Cleanup
    free(output);
    onebit_cleanup(&ctx);
    
    return 0;
} 