#include <onebit/onebit_config.h>
#include <onebit/onebit_error.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

// Global configuration
static OneBitConfig global_config;
static pthread_mutex_t config_mutex = PTHREAD_MUTEX_INITIALIZER;
static bool config_initialized = false;

// Hardware capabilities cache
static uint32_t hardware_capabilities = HW_CAPABILITY_NONE;
static bool capabilities_detected = false;

// Get default configuration
OneBitConfig config_get_default(void) {
    OneBitConfig config;
    
    // Initialize with default values
    config.device_id = ONEBIT_DEFAULT_DEVICE;
    config.memory_pool_size = ONEBIT_DEFAULT_MEMORY_SIZE;
    config.num_threads = ONEBIT_DEFAULT_NUM_THREADS;
    config.batch_size = ONEBIT_DEFAULT_BATCH_SIZE;
    config.max_seq_len = ONEBIT_DEFAULT_MAX_SEQ_LEN;
    config.learning_rate = ONEBIT_DEFAULT_LEARNING_RATE;
    config.weight_decay = ONEBIT_DEFAULT_WEIGHT_DECAY;
    config.epsilon = ONEBIT_DEFAULT_EPSILON;
    config.random_seed = ONEBIT_DEFAULT_RANDOM_SEED;
    config.log_level = ONEBIT_DEFAULT_LOG_LEVEL;
    config.log_file = NULL;
    config.use_cache = true;
    config.use_mixed_precision = false;
    config.use_quantization = true;
    config.prefetch_size = 2;
    config.checkpoint_interval = 1000;
    config.enable_profiling = false;
    config.enable_parallel = true;
    config.gradient_clip = 1.0f;
    config.verbose = 1;
    
    return config;
}

// Initialize global configuration if needed
static void ensure_config_initialized(void) {
    if (!config_initialized) {
        pthread_mutex_lock(&config_mutex);
        if (!config_initialized) {
            global_config = config_get_default();
            config_initialized = true;
        }
        pthread_mutex_unlock(&config_mutex);
    }
}

// Set global configuration
int config_set_global(const OneBitConfig* config) {
    if (!config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&config_mutex);
    memcpy(&global_config, config, sizeof(OneBitConfig));
    config_initialized = true;
    pthread_mutex_unlock(&config_mutex);
    
    return ONEBIT_SUCCESS;
}

// Get global configuration
int config_get_global(OneBitConfig* config) {
    if (!config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ensure_config_initialized();
    
    pthread_mutex_lock(&config_mutex);
    memcpy(config, &global_config, sizeof(OneBitConfig));
    pthread_mutex_unlock(&config_mutex);
    
    return ONEBIT_SUCCESS;
}

// Reset global configuration to defaults
int config_reset_global(void) {
    pthread_mutex_lock(&config_mutex);
    global_config = config_get_default();
    config_initialized = true;
    pthread_mutex_unlock(&config_mutex);
    
    return ONEBIT_SUCCESS;
}

// Detect hardware capabilities
static void detect_hardware_capabilities(void) {
    if (capabilities_detected) {
        return;
    }
    
    pthread_mutex_lock(&config_mutex);
    if (capabilities_detected) {
        pthread_mutex_unlock(&config_mutex);
        return;
    }
    
    uint32_t caps = HW_CAPABILITY_NONE;
    
    // CPU capabilities detection
    #if defined(__x86_64__) || defined(_M_X64)
        // Basic x86-64 support is always available in this case
        
        #if defined(__AVX__)
            caps |= HW_CAPABILITY_AVX;
        #endif
        
        #if defined(__AVX2__)
            caps |= HW_CAPABILITY_AVX2;
        #endif
        
        #if defined(__AVX512F__)
            caps |= HW_CAPABILITY_AVX512;
        #endif
        
        #if defined(__FMA__)
            caps |= HW_CAPABILITY_FMA;
        #endif
    #elif defined(__aarch64__) || defined(_M_ARM64)
        // ARM64 platform
        
        #if defined(__ARM_NEON) || defined(__ARM_NEON__)
            caps |= HW_CAPABILITY_NEON;
        #endif
    #endif
    
    // TODO: Add runtime detection for CUDA, ROCm, etc.
    // This would involve checking for devices and libraries.
    
    // For now, we'll just check compile-time defines
    #if defined(ONEBIT_WITH_CUDA)
        caps |= HW_CAPABILITY_CUDA;
    #endif
    
    #if defined(ONEBIT_WITH_ROCM)
        caps |= HW_CAPABILITY_ROCM;
    #endif
    
    #if defined(ONEBIT_WITH_METAL)
        caps |= HW_CAPABILITY_METAL;
    #endif
    
    #if defined(ONEBIT_WITH_OPENCL)
        caps |= HW_CAPABILITY_OPENCL;
    #endif
    
    #if defined(ONEBIT_WITH_VULKAN)
        caps |= HW_CAPABILITY_VULKAN;
    #endif
    
    // Binary ops support
    #if defined(ONEBIT_WITH_BINARY_OPS)
        caps |= HW_CAPABILITY_BINARY;
    #endif
    
    hardware_capabilities = caps;
    capabilities_detected = true;
    
    pthread_mutex_unlock(&config_mutex);
}

// Get hardware capabilities
uint32_t config_get_hardware_capabilities(void) {
    detect_hardware_capabilities();
    return hardware_capabilities;
}

// Check if a hardware capability is available
bool config_has_capability(HardwareCapability capability) {
    detect_hardware_capabilities();
    return (hardware_capabilities & capability) != 0;
}

// Print configuration
void config_print(const OneBitConfig* config) {
    if (!config) {
        return;
    }
    
    fprintf(stderr, "OneBit Configuration:\n");
    fprintf(stderr, "  - Version: %s\n", ONEBIT_VERSION_STRING);
    fprintf(stderr, "  - Device ID: %d\n", config->device_id);
    fprintf(stderr, "  - Memory Pool: %zu bytes\n", config->memory_pool_size);
    fprintf(stderr, "  - Threads: %zu\n", config->num_threads);
    fprintf(stderr, "  - Batch Size: %zu\n", config->batch_size);
    fprintf(stderr, "  - Max Sequence Length: %zu\n", config->max_seq_len);
    fprintf(stderr, "  - Learning Rate: %.6f\n", config->learning_rate);
    fprintf(stderr, "  - Weight Decay: %.6f\n", config->weight_decay);
    fprintf(stderr, "  - Epsilon: %.6f\n", config->epsilon);
    fprintf(stderr, "  - Random Seed: %u\n", config->random_seed);
    fprintf(stderr, "  - Log Level: %d\n", config->log_level);
    fprintf(stderr, "  - Log File: %s\n", config->log_file ? config->log_file : "stdout");
    fprintf(stderr, "  - Use Cache: %s\n", config->use_cache ? "true" : "false");
    fprintf(stderr, "  - Mixed Precision: %s\n", config->use_mixed_precision ? "true" : "false");
    fprintf(stderr, "  - Quantization: %s\n", config->use_quantization ? "true" : "false");
    fprintf(stderr, "  - Prefetch Size: %zu\n", config->prefetch_size);
    fprintf(stderr, "  - Checkpoint Interval: %zu\n", config->checkpoint_interval);
    fprintf(stderr, "  - Profiling: %s\n", config->enable_profiling ? "true" : "false");
    fprintf(stderr, "  - Parallel Execution: %s\n", config->enable_parallel ? "true" : "false");
    fprintf(stderr, "  - Gradient Clip: %.2f\n", config->gradient_clip);
    fprintf(stderr, "  - Verbosity: %d\n", config->verbose);
    
    // Print hardware capabilities
    fprintf(stderr, "Hardware Capabilities:\n");
    uint32_t caps = config_get_hardware_capabilities();
    
    if (caps & HW_CAPABILITY_AVX) fprintf(stderr, "  - AVX: Supported\n");
    if (caps & HW_CAPABILITY_AVX2) fprintf(stderr, "  - AVX2: Supported\n");
    if (caps & HW_CAPABILITY_AVX512) fprintf(stderr, "  - AVX-512: Supported\n");
    if (caps & HW_CAPABILITY_FMA) fprintf(stderr, "  - FMA: Supported\n");
    if (caps & HW_CAPABILITY_NEON) fprintf(stderr, "  - NEON: Supported\n");
    if (caps & HW_CAPABILITY_CUDA) fprintf(stderr, "  - CUDA: Supported\n");
    if (caps & HW_CAPABILITY_ROCM) fprintf(stderr, "  - ROCm: Supported\n");
    if (caps & HW_CAPABILITY_METAL) fprintf(stderr, "  - Metal: Supported\n");
    if (caps & HW_CAPABILITY_OPENCL) fprintf(stderr, "  - OpenCL: Supported\n");
    if (caps & HW_CAPABILITY_VULKAN) fprintf(stderr, "  - Vulkan: Supported\n");
    if (caps & HW_CAPABILITY_BINARY) fprintf(stderr, "  - Binary Ops: Supported\n");
}

// Set a configuration parameter by name
int config_set_param(OneBitConfig* config, const char* name, const char* value) {
    if (!config || !name || !value) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    if (strcmp(name, "device_id") == 0) {
        config->device_id = atoi(value);
    } else if (strcmp(name, "memory_pool_size") == 0) {
        config->memory_pool_size = (size_t)atol(value);
    } else if (strcmp(name, "num_threads") == 0) {
        config->num_threads = (size_t)atol(value);
    } else if (strcmp(name, "batch_size") == 0) {
        config->batch_size = (size_t)atol(value);
    } else if (strcmp(name, "max_seq_len") == 0) {
        config->max_seq_len = (size_t)atol(value);
    } else if (strcmp(name, "learning_rate") == 0) {
        config->learning_rate = (float)atof(value);
    } else if (strcmp(name, "weight_decay") == 0) {
        config->weight_decay = (float)atof(value);
    } else if (strcmp(name, "epsilon") == 0) {
        config->epsilon = (float)atof(value);
    } else if (strcmp(name, "random_seed") == 0) {
        config->random_seed = (uint32_t)atol(value);
    } else if (strcmp(name, "log_level") == 0) {
        config->log_level = (LogLevel)atoi(value);
    } else if (strcmp(name, "log_file") == 0) {
        config->log_file = strdup(value);
    } else if (strcmp(name, "use_cache") == 0) {
        config->use_cache = (strcmp(value, "true") == 0 || strcmp(value, "1") == 0);
    } else if (strcmp(name, "use_mixed_precision") == 0) {
        config->use_mixed_precision = (strcmp(value, "true") == 0 || strcmp(value, "1") == 0);
    } else if (strcmp(name, "use_quantization") == 0) {
        config->use_quantization = (strcmp(value, "true") == 0 || strcmp(value, "1") == 0);
    } else if (strcmp(name, "prefetch_size") == 0) {
        config->prefetch_size = (size_t)atol(value);
    } else if (strcmp(name, "checkpoint_interval") == 0) {
        config->checkpoint_interval = (size_t)atol(value);
    } else if (strcmp(name, "enable_profiling") == 0) {
        config->enable_profiling = (strcmp(value, "true") == 0 || strcmp(value, "1") == 0);
    } else if (strcmp(name, "enable_parallel") == 0) {
        config->enable_parallel = (strcmp(value, "true") == 0 || strcmp(value, "1") == 0);
    } else if (strcmp(name, "gradient_clip") == 0) {
        config->gradient_clip = (float)atof(value);
    } else if (strcmp(name, "verbose") == 0) {
        config->verbose = atoi(value);
    } else {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    return ONEBIT_SUCCESS;
}

// Get a configuration parameter by name
int config_get_param(const OneBitConfig* config, const char* name, char* value, size_t max_len) {
    if (!config || !name || !value || max_len == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    if (strcmp(name, "device_id") == 0) {
        snprintf(value, max_len, "%d", config->device_id);
    } else if (strcmp(name, "memory_pool_size") == 0) {
        snprintf(value, max_len, "%zu", config->memory_pool_size);
    } else if (strcmp(name, "num_threads") == 0) {
        snprintf(value, max_len, "%zu", config->num_threads);
    } else if (strcmp(name, "batch_size") == 0) {
        snprintf(value, max_len, "%zu", config->batch_size);
    } else if (strcmp(name, "max_seq_len") == 0) {
        snprintf(value, max_len, "%zu", config->max_seq_len);
    } else if (strcmp(name, "learning_rate") == 0) {
        snprintf(value, max_len, "%.6f", config->learning_rate);
    } else if (strcmp(name, "weight_decay") == 0) {
        snprintf(value, max_len, "%.6f", config->weight_decay);
    } else if (strcmp(name, "epsilon") == 0) {
        snprintf(value, max_len, "%.6f", config->epsilon);
    } else if (strcmp(name, "random_seed") == 0) {
        snprintf(value, max_len, "%u", config->random_seed);
    } else if (strcmp(name, "log_level") == 0) {
        snprintf(value, max_len, "%d", config->log_level);
    } else if (strcmp(name, "log_file") == 0) {
        if (config->log_file) {
            snprintf(value, max_len, "%s", config->log_file);
        } else {
            snprintf(value, max_len, "stdout");
        }
    } else if (strcmp(name, "use_cache") == 0) {
        snprintf(value, max_len, "%s", config->use_cache ? "true" : "false");
    } else if (strcmp(name, "use_mixed_precision") == 0) {
        snprintf(value, max_len, "%s", config->use_mixed_precision ? "true" : "false");
    } else if (strcmp(name, "use_quantization") == 0) {
        snprintf(value, max_len, "%s", config->use_quantization ? "true" : "false");
    } else if (strcmp(name, "prefetch_size") == 0) {
        snprintf(value, max_len, "%zu", config->prefetch_size);
    } else if (strcmp(name, "checkpoint_interval") == 0) {
        snprintf(value, max_len, "%zu", config->checkpoint_interval);
    } else if (strcmp(name, "enable_profiling") == 0) {
        snprintf(value, max_len, "%s", config->enable_profiling ? "true" : "false");
    } else if (strcmp(name, "enable_parallel") == 0) {
        snprintf(value, max_len, "%s", config->enable_parallel ? "true" : "false");
    } else if (strcmp(name, "gradient_clip") == 0) {
        snprintf(value, max_len, "%.2f", config->gradient_clip);
    } else if (strcmp(name, "verbose") == 0) {
        snprintf(value, max_len, "%d", config->verbose);
    } else {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    return ONEBIT_SUCCESS;
}

// Parse configuration from command line arguments
int config_parse_args(int argc, char** argv, OneBitConfig* config) {
    if (!config || !argv) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Start with default config
    *config = config_get_default();
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-') {
            // This is an option
            const char* option = argv[i] + 1;
            
            // Handle --option=value format
            char* value = strchr(option, '=');
            if (value) {
                // Split the option at the '=' character
                *value = '\0';
                value++;
                
                // If option begins with '-', it's a long option (--option)
                if (option[0] == '-') {
                    option++;
                }
                
                // Set the parameter
                int result = config_set_param(config, option, value);
                if (result != ONEBIT_SUCCESS) {
                    fprintf(stderr, "Invalid option: %s\n", option);
                }
                
                // Restore the '=' for future reference
                value[-1] = '=';
            } else if (i + 1 < argc) {
                // Handle -o value format
                
                // If option begins with '-', it's a long option (--option)
                if (option[0] == '-') {
                    option++;
                }
                
                int result = config_set_param(config, option, argv[i + 1]);
                if (result != ONEBIT_SUCCESS) {
                    fprintf(stderr, "Invalid option: %s\n", option);
                } else {
                    // Skip the value in the next iteration
                    i++;
                }
            }
        }
    }
    
    return ONEBIT_SUCCESS;
}

// Load configuration from a JSON file
int config_load_file(const char* path, OneBitConfig* config) {
    if (!path || !config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // TODO: Implement JSON file loading
    // This would typically use a JSON parsing library
    
    // For now, just use the default configuration
    *config = config_get_default();
    
    return ONEBIT_ERROR_NOT_IMPLEMENTED;
}

// Save configuration to a JSON file
int config_save_file(const char* path, const OneBitConfig* config) {
    if (!path || !config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // TODO: Implement JSON file saving
    // This would typically use a JSON generation library
    
    return ONEBIT_ERROR_NOT_IMPLEMENTED;
} 