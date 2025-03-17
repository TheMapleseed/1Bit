#include <onebit/onebit.h>
#include <onebit/onebit_model.h>
#include <onebit/onebit_compute.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Default configuration
OneBitConfig onebit_default_config(void) {
    OneBitConfig config;
    
    // Model configuration
    config.vocab_size = 50000;
    config.hidden_size = 768;
    config.num_layers = 12;
    config.num_heads = 12;
    config.intermediate_size = 3072;
    config.max_position = 1024;
    
    // Training configuration
    config.attention_dropout = 0.1f;
    config.hidden_dropout = 0.1f;
    config.embedding_dropout = 0.1f;
    config.layer_norm_eps = 1e-5f;
    
    // Runtime configuration
    config.use_cuda = false;
    config.use_rocm = false;
    config.use_metal = false;
    config.device_id = 0;
    config.use_quantization = true;
    config.num_threads = 4;
    
    // Memory configuration
    config.memory_pool_size = 512 * 1024 * 1024;  // 512MB
    config.use_memory_pool = true;
    
    // Tokenizer configuration
    config.vocab_file = NULL;
    config.add_special_tokens = true;
    
    return config;
}

// Initialize the OneBit context
int onebit_init(OneBitContext* ctx, const OneBitConfig* config) {
    if (!ctx || !config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Initialize fields to NULL/0 first
    memset(ctx, 0, sizeof(OneBitContext));
    
    // Copy the configuration
    memcpy(&ctx->config, config, sizeof(OneBitConfig));
    
    // Initialize compute context
    ctx->compute = malloc(sizeof(ComputeContext));
    if (!ctx->compute) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    int result = compute_init(ctx->compute, config->use_cuda, config->use_rocm, config->use_metal);
    if (result != ONEBIT_SUCCESS) {
        free(ctx->compute);
        return result;
    }
    
    // Initialize memory pool if requested
    if (config->use_memory_pool) {
        // Memory pool initialization would go here
        // For now, just simulate success
    }
    
    ctx->is_initialized = true;
    return ONEBIT_SUCCESS;
}

// Clean up the OneBit context
void onebit_cleanup(OneBitContext* ctx) {
    if (!ctx) return;
    
    // Clean up model if it exists
    if (ctx->model) {
        model_cleanup(ctx->model);
        free(ctx->model);
        ctx->model = NULL;
    }
    
    // Clean up tokenizer if it exists
    if (ctx->tokenizer) {
        // Tokenizer cleanup would go here
        free(ctx->tokenizer);
        ctx->tokenizer = NULL;
    }
    
    // Clean up memory pool if it exists
    if (ctx->pool) {
        // Memory pool cleanup would go here
        free(ctx->pool);
        ctx->pool = NULL;
    }
    
    // Clean up compute context
    if (ctx->compute) {
        compute_cleanup(ctx->compute);
        free(ctx->compute);
        ctx->compute = NULL;
    }
    
    ctx->is_initialized = false;
}

// Load a model from file
int onebit_load_model(OneBitContext* ctx, const char* model_path) {
    if (!ctx || !model_path) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    if (!ctx->is_initialized) {
        return ONEBIT_ERROR_RUNTIME;
    }
    
    // Create model if it doesn't exist
    if (!ctx->model) {
        ctx->model = malloc(sizeof(OneBitModel));
        if (!ctx->model) {
            return ONEBIT_ERROR_MEMORY;
        }
        
        int result = model_init(ctx->model, &ctx->config);
        if (result != ONEBIT_SUCCESS) {
            free(ctx->model);
            ctx->model = NULL;
            return result;
        }
    }
    
    // Load model weights
    return model_load_weights(ctx->model, model_path);
}

// Generate text from a prompt
int onebit_generate(OneBitContext* ctx, const char* prompt, 
                   char* output, int max_length,
                   float temperature, float top_p) {
    if (!ctx || !prompt || !output || max_length <= 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    if (!ctx->is_initialized || !ctx->model) {
        return ONEBIT_ERROR_RUNTIME;
    }
    
    // TODO: Implement actual tokenization and generation
    // For this basic implementation, we'll just copy the prompt to output
    // to demonstrate the API structure
    
    strncpy(output, prompt, max_length - 1);
    output[max_length - 1] = '\0';
    
    // Append a placeholder generated text
    const char* generated = " [This is placeholder generated text from the OneBit library]";
    size_t prompt_len = strlen(output);
    size_t generated_len = strlen(generated);
    
    if (prompt_len + generated_len < max_length - 1) {
        strcat(output, generated);
    }
    
    return ONEBIT_SUCCESS;
}

// Get the version string
const char* onebit_version(void) {
    return ONEBIT_VERSION;
}

// Get error string from error code
const char* onebit_error_string(int error_code) {
    switch (error_code) {
        case ONEBIT_SUCCESS:
            return "Success";
        case ONEBIT_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case ONEBIT_ERROR_MEMORY:
            return "Memory allocation error";
        case ONEBIT_ERROR_IO:
            return "I/O error";
        case ONEBIT_ERROR_RUNTIME:
            return "Runtime error";
        case ONEBIT_ERROR_THREAD:
            return "Threading error";
        case ONEBIT_ERROR_NOT_SUPPORTED:
            return "Operation not supported";
        case ONEBIT_ERROR_CUDA:
            return "CUDA error";
        case ONEBIT_ERROR_METAL:
            return "Metal error";
        case ONEBIT_ERROR_ROCM:
            return "ROCm error";
        default:
            return "Unknown error";
    }
}

// Print configuration
void onebit_print_config(const OneBitConfig* config) {
    if (!config) return;
    
    printf("OneBit Configuration:\n");
    printf("  Model:\n");
    printf("    vocab_size: %d\n", config->vocab_size);
    printf("    hidden_size: %d\n", config->hidden_size);
    printf("    num_layers: %d\n", config->num_layers);
    printf("    num_heads: %d\n", config->num_heads);
    printf("    intermediate_size: %d\n", config->intermediate_size);
    printf("    max_position: %d\n", config->max_position);
    
    printf("  Runtime:\n");
    printf("    use_cuda: %s\n", config->use_cuda ? "Yes" : "No");
    printf("    use_rocm: %s\n", config->use_rocm ? "Yes" : "No");
    printf("    use_metal: %s\n", config->use_metal ? "Yes" : "No");
    printf("    device_id: %d\n", config->device_id);
    printf("    num_threads: %d\n", config->num_threads);
    printf("    use_quantization: %s\n", config->use_quantization ? "Yes" : "No");
    
    printf("  Memory:\n");
    printf("    use_memory_pool: %s\n", config->use_memory_pool ? "Yes" : "No");
    printf("    memory_pool_size: %zu MB\n", config->memory_pool_size / (1024 * 1024));
}

// Hardware capability checks
bool onebit_has_avx512(void) {
    // This should be implemented with actual CPU feature detection
    // For now, return a placeholder value based on compile-time flags
#ifdef ONEBIT_ENABLE_AVX512
    return true;
#else
    return false;
#endif
}

bool onebit_has_avx2(void) {
#ifdef ONEBIT_ENABLE_AVX2
    return true;
#else
    return false;
#endif
}

bool onebit_has_sse4(void) {
#ifdef ONEBIT_ENABLE_SSE4
    return true;
#else
    return false;
#endif
}

// Get device count
int onebit_get_device_count(void) {
    // This should call into the actual device enumeration
    // For now, return a placeholder value
    return compute_get_device_count();
}

// Get device info
int onebit_get_device_info(int device_id, char* name, size_t* memory) {
    // This should call into the actual device query
    // For now, return placeholder values
    if (name) {
        strcpy(name, "Placeholder GPU Device");
    }
    
    if (memory) {
        *memory = 8 * 1024 * 1024 * 1024ULL;  // 8GB
    }
    
    return ONEBIT_SUCCESS;
} 