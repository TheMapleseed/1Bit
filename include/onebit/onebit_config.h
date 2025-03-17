/**
 * @file onebit_config.h
 * @brief Configuration parameters and settings for OneBit framework
 * @author OneBit Team
 * @version 0.1.0
 * @date 2023-03-17
 */

#ifndef ONEBIT_onebit_config_H
#define ONEBIT_onebit_config_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Major version number
 */
#define ONEBIT_VERSION_MAJOR 0

/**
 * @brief Minor version number
 */
#define ONEBIT_VERSION_MINOR 1

/**
 * @brief Patch version number
 */
#define ONEBIT_VERSION_PATCH 0

/**
 * @brief Version string
 */
#define ONEBIT_VERSION_STRING "0.1.0"

/**
 * @brief Default compute device to use (-1 for CPU)
 */
#define ONEBIT_DEFAULT_DEVICE -1

/**
 * @brief Default memory pool size (1GB)
 */
#define ONEBIT_DEFAULT_MEMORY_SIZE (1024 * 1024 * 1024)

/**
 * @brief Default number of threads for parallel operations
 */
#define ONEBIT_DEFAULT_NUM_THREADS 4

/**
 * @brief Default batch size for training and inference
 */
#define ONEBIT_DEFAULT_BATCH_SIZE 32

/**
 * @brief Default learning rate for training
 */
#define ONEBIT_DEFAULT_LEARNING_RATE 0.001f

/**
 * @brief Default weight decay for training
 */
#define ONEBIT_DEFAULT_WEIGHT_DECAY 0.01f

/**
 * @brief Default epsilon value for numerical stability
 */
#define ONEBIT_DEFAULT_EPSILON 1e-5f

/**
 * @brief Default random seed
 */
#define ONEBIT_DEFAULT_RANDOM_SEED 42

/**
 * @brief Default maximum sequence length
 */
#define ONEBIT_DEFAULT_MAX_SEQ_LEN 2048

/**
 * @brief Default log level
 */
#define ONEBIT_DEFAULT_LOG_LEVEL 2

/**
 * @brief Maximum debug function name length
 */
#define ONEBIT_MAX_FUNC_NAME 64

/**
 * @brief Maximum debug file name length
 */
#define ONEBIT_MAX_FILE_NAME 256

/**
 * @brief Maximum error message length
 */
#define ONEBIT_MAX_ERROR_MSG 1024

/**
 * @brief Log levels
 */
typedef enum {
    LOG_LEVEL_NONE = 0,    /**< No logging */
    LOG_LEVEL_ERROR = 1,   /**< Error messages only */
    LOG_LEVEL_WARN = 2,    /**< Warning and error messages */
    LOG_LEVEL_INFO = 3,    /**< Info, warning, and error messages */
    LOG_LEVEL_DEBUG = 4,   /**< Debug, info, warning, and error messages */
    LOG_LEVEL_TRACE = 5    /**< Trace, debug, info, warning, and error messages */
} LogLevel;

/**
 * @brief Runtime configuration structure
 */
typedef struct {
    int device_id;              /**< Compute device ID (-1 for CPU) */
    size_t memory_pool_size;    /**< Memory pool size in bytes */
    size_t num_threads;         /**< Number of threads for parallel operations */
    size_t batch_size;          /**< Batch size for training/inference */
    size_t max_seq_len;         /**< Maximum sequence length */
    float learning_rate;        /**< Learning rate for training */
    float weight_decay;         /**< Weight decay for regularization */
    float epsilon;              /**< Epsilon value for numerical stability */
    uint32_t random_seed;       /**< Random seed */
    LogLevel log_level;         /**< Logging level */
    const char* log_file;       /**< Log file path (NULL for stdout) */
    bool use_cache;             /**< Whether to use cache for optimization */
    bool use_mixed_precision;   /**< Whether to use mixed precision */
    bool use_quantization;      /**< Whether to use quantization */
    size_t prefetch_size;       /**< Prefetch size for data loading */
    size_t checkpoint_interval; /**< Checkpoint interval in steps */
    bool enable_profiling;      /**< Whether to enable profiling */
    bool enable_parallel;       /**< Whether to enable parallel execution */
    float gradient_clip;        /**< Gradient clipping value */
    int verbose;                /**< Verbosity level (0-3) */
} OneBitConfig;

/**
 * @brief Hardware capability flags
 */
typedef enum {
    HW_CAPABILITY_NONE = 0,       /**< No special capabilities */
    HW_CAPABILITY_AVX = 1,        /**< AVX support */
    HW_CAPABILITY_AVX2 = 2,       /**< AVX2 support */
    HW_CAPABILITY_AVX512 = 4,     /**< AVX-512 support */
    HW_CAPABILITY_FMA = 8,        /**< FMA support */
    HW_CAPABILITY_NEON = 16,      /**< ARM NEON support */
    HW_CAPABILITY_CUDA = 32,      /**< CUDA support */
    HW_CAPABILITY_ROCM = 64,      /**< ROCm support */
    HW_CAPABILITY_METAL = 128,    /**< Metal support */
    HW_CAPABILITY_OPENCL = 256,   /**< OpenCL support */
    HW_CAPABILITY_VULKAN = 512,   /**< Vulkan compute support */
    HW_CAPABILITY_BINARY = 1024   /**< Binary operation support */
} HardwareCapability;

/**
 * @brief Get default configuration
 * @return Default configuration structure
 */
OneBitConfig config_get_default(void);

/**
 * @brief Set global configuration
 * @param config Configuration to set
 * @return Error code (0 for success)
 */
int config_set_global(const OneBitConfig* config);

/**
 * @brief Get global configuration
 * @param config Pointer to store the configuration
 * @return Error code (0 for success)
 */
int config_get_global(OneBitConfig* config);

/**
 * @brief Reset global configuration to defaults
 * @return Error code (0 for success)
 */
int config_reset_global(void);

/**
 * @brief Load configuration from a JSON file
 * @param path Path to the configuration file
 * @param config Pointer to store the loaded configuration
 * @return Error code (0 for success)
 */
int config_load_file(const char* path, OneBitConfig* config);

/**
 * @brief Save configuration to a JSON file
 * @param path Path to save the configuration file
 * @param config Configuration to save
 * @return Error code (0 for success)
 */
int config_save_file(const char* path, const OneBitConfig* config);

/**
 * @brief Parse configuration from command line arguments
 * @param argc Argument count
 * @param argv Argument values
 * @param config Pointer to store the parsed configuration
 * @return Error code (0 for success)
 */
int config_parse_args(int argc, char** argv, OneBitConfig* config);

/**
 * @brief Print configuration to stderr
 * @param config Configuration to print
 */
void config_print(const OneBitConfig* config);

/**
 * @brief Get hardware capabilities
 * @return Bitfield of hardware capabilities
 */
uint32_t config_get_hardware_capabilities(void);

/**
 * @brief Set a configuration parameter by name
 * @param config Configuration to modify
 * @param name Parameter name
 * @param value Parameter value as string
 * @return Error code (0 for success)
 */
int config_set_param(OneBitConfig* config, const char* name, const char* value);

/**
 * @brief Get a configuration parameter by name
 * @param config Configuration to query
 * @param name Parameter name
 * @param value Buffer to store the string value
 * @param max_len Maximum length of the value buffer
 * @return Error code (0 for success)
 */
int config_get_param(const OneBitConfig* config, const char* name, char* value, size_t max_len);

/**
 * @brief Check if a hardware capability is available
 * @param capability Capability to check
 * @return true if the capability is available, false otherwise
 */
bool config_has_capability(HardwareCapability capability);

#ifdef __cplusplus
}
#endif

#endif /* ONEBIT_CONFIG_H */ 