#include <onebit/onebit.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
 * @brief Example application for OneBit model quantization
 * 
 * This example loads a full-precision model and quantizes it to
 * binary weights using the OneBit library.
 */
int main(int argc, char** argv) {
    // Check arguments
    if (argc < 3) {
        printf("Usage: %s <input_model_path> <output_model_path> [quant_type]\n", argv[0]);
        printf("Quantization types:\n");
        printf("  0 - Binary (1-bit)\n");
        printf("  1 - INT4 (4-bit)\n");
        printf("  2 - INT8 (8-bit)\n");
        return 1;
    }
    
    const char* input_path = argv[1];
    const char* output_path = argv[2];
    int quant_type = (argc > 3) ? atoi(argv[3]) : 0;  // Default to binary
    
    ParamType target_type;
    switch (quant_type) {
        case 0: target_type = PARAM_TYPE_BINARY; break;
        case 1: target_type = PARAM_TYPE_INT4; break;
        case 2: target_type = PARAM_TYPE_INT8; break;
        default:
            printf("Invalid quantization type: %d\n", quant_type);
            return 1;
    }
    
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
    printf("Loading model from %s...\n", input_path);
    result = onebit_load_model(&ctx, input_path);
    if (result != ONEBIT_SUCCESS) {
        printf("Error loading model: %s\n", onebit_error_string(result));
        onebit_cleanup(&ctx);
        return 1;
    }
    
    // Get model size before quantization
    size_t orig_size = model_get_num_parameters(ctx.model);
    printf("Original model size: %zu parameters\n", orig_size);
    
    // Start timing
    printf("Quantizing model to ");
    switch (target_type) {
        case PARAM_TYPE_BINARY: printf("binary (1-bit)"); break;
        case PARAM_TYPE_INT4: printf("INT4 (4-bit)"); break;
        case PARAM_TYPE_INT8: printf("INT8 (8-bit)"); break;
        default: printf("unknown type");
    }
    printf("...\n");
    
    clock_t start = clock();
    
    // Quantize the model
    result = model_quantize(ctx.model, target_type);
    if (result != ONEBIT_SUCCESS) {
        printf("Error quantizing model: %s\n", onebit_error_string(result));
        onebit_cleanup(&ctx);
        return 1;
    }
    
    // End timing
    clock_t end = clock();
    float elapsed = (float)(end - start) / CLOCKS_PER_SEC;
    
    // Get model size after quantization
    size_t quant_size = model_get_num_parameters(ctx.model);
    printf("Quantized model size: %zu parameters\n", quant_size);
    printf("Compression ratio: %.2fx\n", (float)orig_size / quant_size);
    printf("Quantization completed in %.2f seconds\n", elapsed);
    
    // Save the quantized model
    printf("Saving quantized model to %s...\n", output_path);
    result = model_save_weights(ctx.model, output_path);
    if (result != ONEBIT_SUCCESS) {
        printf("Error saving model: %s\n", onebit_error_string(result));
        onebit_cleanup(&ctx);
        return 1;
    }
    
    // Optional: verify the model
    printf("Verifying quantized model...\n");
    result = model_verify_weights(ctx.model);
    if (result != ONEBIT_SUCCESS) {
        printf("Model verification failed: %s\n", onebit_error_string(result));
    } else {
        printf("Model verification passed\n");
    }
    
    // Cleanup
    onebit_cleanup(&ctx);
    
    printf("Model quantized and saved successfully\n");
    return 0;
} 