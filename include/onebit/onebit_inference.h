/**
 * @file onebit_inference.h
 * @brief High-performance inference and online learning system
 */

#ifndef ONEBIT_onebit_inference_H
#define ONEBIT_onebit_inference_H

#include "onebit_model.h"
#include "onebit_train.h"
#include <stdint.h>
#include <stdbool.h>

// Forward declarations
typedef struct ModelContext ModelContext;
typedef struct TokenizerContext TokenizerContext;
typedef struct InferenceContext InferenceContext;

/**
 * @brief Configuration for inference
 */
typedef struct {
    // General inference parameters
    float temperature;            // Temperature for sampling (higher = more random)
    float top_p;                  // Top-p sampling threshold (0-1)
    size_t top_k;                 // Top-k sampling threshold (0 = disabled)
    
    // Sequence constraints
    size_t max_seq_len;           // Maximum sequence length for generated text
    size_t min_seq_len;           // Minimum sequence length for generated text
    
    // Resource constraints
    size_t batch_size;            // Batch size for parallel inference
    bool use_int8;                // Use int8 quantization for inference
    bool use_fp16;                // Use fp16 for inference
    bool use_kv_cache;            // Use KV cache optimization
    
    // Conditional generation
    bool use_logit_bias;          // Apply token biases
    float* logit_bias;            // Per-token bias values
    
    // System parameters
    int num_threads;              // Number of CPU threads to use
    bool pin_memory;              // Pin memory for optimal cache usage
    bool stream_output;           // Stream tokens as they are generated
    
    // Online learning
    bool enable_online_learning;  // Enable online learning during inference
    OnlineLearningConfig online_learning_config;  // Online learning configuration
} InferenceConfig;

/**
 * @brief Initialize the inference engine
 * 
 * @param ctx Output parameter for the created context
 * @param model The model to use for inference
 * @param tokenizer Tokenizer for text processing
 * @param config Inference configuration
 * @return int Error code (0 = success)
 */
int inference_init(InferenceContext** ctx, const ModelContext* model,
                  const TokenizerContext* tokenizer,
                  const InferenceConfig* config);

/**
 * @brief Free resources associated with inference context
 * 
 * @param ctx Inference context to clean up
 */
void inference_cleanup(InferenceContext* ctx);

/**
 * @brief Generate text from a prompt
 * 
 * @param ctx Inference context
 * @param prompt Input prompt text
 * @param output Buffer for output text
 * @param output_size Size of output buffer on input, number of bytes written on output
 * @param max_tokens Maximum number of tokens to generate
 * @return int Error code (0 = success)
 */
int inference_generate(InferenceContext* ctx, const char* prompt,
                      char* output, size_t* output_size,
                      size_t max_tokens);

/**
 * @brief Generate the next token given input tokens
 * 
 * @param ctx Inference context
 * @param input_tokens Input token sequence
 * @param num_tokens Number of input tokens
 * @param next_token Output parameter for the next token
 * @return int Error code (0 = success)
 */
int inference_next_token(InferenceContext* ctx, const uint32_t* input_tokens,
                        size_t num_tokens, uint32_t* next_token);

/**
 * @brief Get raw logits for next token prediction
 * 
 * @param ctx Inference context
 * @param input_tokens Input token sequence
 * @param num_tokens Number of input tokens
 * @param logits Output buffer for logits
 * @param vocab_size Size of vocabulary (and logits buffer)
 * @return int Error code (0 = success)
 */
int inference_get_logits(InferenceContext* ctx, const uint32_t* input_tokens,
                        size_t num_tokens, float* logits, size_t vocab_size);

/**
 * @brief Update inference parameters
 * 
 * @param ctx Inference context
 * @param config New configuration
 * @return int Error code (0 = success)
 */
int inference_set_params(InferenceContext* ctx, const InferenceConfig* config);

/**
 * @brief Reset inference state (clears KV cache and online learning state)
 * 
 * @param ctx Inference context
 */
void inference_reset(InferenceContext* ctx);

/**
 * @brief Check if inference context is initialized
 * 
 * @param ctx Inference context
 * @return bool True if initialized
 */
bool inference_is_initialized(const InferenceContext* ctx);

/**
 * @brief Get the last error message
 * 
 * @param ctx Inference context
 * @return const char* Error message string
 */
const char* inference_get_error(const InferenceContext* ctx);

/**
 * @brief Provide explicit feedback for online learning
 * 
 * @param ctx Inference context
 * @param target_output The expected output for the previous input
 * @param loss Output parameter for learning loss (optional, can be NULL)
 * @return int Error code (0 = success)
 */
int inference_provide_feedback(InferenceContext* ctx, const float* target_output, float* loss);

/**
 * @brief Get statistics about online learning during inference
 * 
 * @param ctx Inference context
 * @param num_updates Output parameter for number of updates performed
 * @param avg_loss Output parameter for average loss across updates
 * @return int Error code (0 = success)
 */
int inference_get_online_learning_stats(InferenceContext* ctx, int* num_updates, float* avg_loss);

#endif // ONEBIT_INFERENCE_H 