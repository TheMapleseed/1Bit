#include "onebit/onebit_inference.h"
#include "onebit/onebit_error.h"
#include "onebit/onebit_compute_opt.h"
#include "onebit/onebit_cache_opt.h"
#include "onebit/onebit_train.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Internal inference context structure
struct InferenceContext {
    const Model* model;
    const TokenizerContext* tokenizer;
    InferenceConfig config;
    
    // KV cache
    CacheContext kv_cache;
    
    // Intermediate buffers
    float* attention_buffer;
    float* ffn_buffer;
    float* logits_buffer;
    
    // Online learning support
    bool online_learning_enabled;
    uint64_t inference_count;
    float* previous_input;
    float* expected_output;
    size_t input_size;
    size_t output_size;
    
    // State tracking
    size_t current_seq_len;
    char error_msg[256];
    bool initialized;
};

// Helper function for sampling from logits
static uint32_t sample_token(const float* logits, size_t vocab_size,
                           float temperature, float top_p, size_t top_k) {
    // Apply temperature
    float* probs = malloc(vocab_size * sizeof(float));
    if (!probs) return 0;
    
    float max_logit = logits[0];
    for (size_t i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_logit) / temperature);
        sum += probs[i];
    }
    
    // Normalize
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }
    
    // Apply top-k if specified
    if (top_k > 0 && top_k < vocab_size) {
        // Find kth largest probability
        float kth_prob = 0.0f;
        for (size_t i = 0; i < vocab_size; i++) {
            size_t count = 0;
            for (size_t j = 0; j < vocab_size; j++) {
                if (probs[j] > probs[i]) count++;
            }
            if (count == top_k - 1) {
                kth_prob = probs[i];
                break;
            }
        }
        
        // Zero out below top-k
        sum = 0.0f;
        for (size_t i = 0; i < vocab_size; i++) {
            if (probs[i] < kth_prob) probs[i] = 0.0f;
            sum += probs[i];
        }
        
        // Renormalize
        for (size_t i = 0; i < vocab_size; i++) {
            probs[i] /= sum;
        }
    }
    
    // Apply nucleus (top-p) sampling
    if (top_p > 0.0f && top_p < 1.0f) {
        // Sort probabilities
        size_t* indices = malloc(vocab_size * sizeof(size_t));
        if (!indices) {
            free(probs);
            return 0;
        }
        
        for (size_t i = 0; i < vocab_size; i++) indices[i] = i;
        
        for (size_t i = 0; i < vocab_size - 1; i++) {
            for (size_t j = i + 1; j < vocab_size; j++) {
                if (probs[indices[j]] > probs[indices[i]]) {
                    size_t temp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = temp;
                }
            }
        }
        
        // Find cutoff for top-p
        float cumsum = 0.0f;
        size_t cutoff = vocab_size;
        for (size_t i = 0; i < vocab_size; i++) {
            cumsum += probs[indices[i]];
            if (cumsum > top_p) {
                cutoff = i + 1;
                break;
            }
        }
        
        // Zero out tokens beyond cutoff
        for (size_t i = cutoff; i < vocab_size; i++) {
            probs[indices[i]] = 0.0f;
        }
        
        free(indices);
        
        // Renormalize
        sum = 0.0f;
        for (size_t i = 0; i < vocab_size; i++) {
            sum += probs[i];
        }
        for (size_t i = 0; i < vocab_size; i++) {
            probs[i] /= sum;
        }
    }
    
    // Sample from distribution
    float r = (float)rand() / RAND_MAX;
    float cdf = 0.0f;
    uint32_t sampled_token = 0;
    
    for (size_t i = 0; i < vocab_size; i++) {
        cdf += probs[i];
        if (r <= cdf) {
            sampled_token = (uint32_t)i;
            break;
        }
    }
    
    free(probs);
    return sampled_token;
}

int inference_init(InferenceContext** ctx, const Model* model,
                  const TokenizerContext* tokenizer,
                  const InferenceConfig* config) {
    if (!ctx || !model || !config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    InferenceContext* new_ctx = (InferenceContext*)malloc(sizeof(InferenceContext));
    if (!new_ctx) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Clear struct
    memset(new_ctx, 0, sizeof(InferenceContext));
    
    // Copy basic settings
    new_ctx->model = model;
    new_ctx->tokenizer = tokenizer;
    memcpy(&new_ctx->config, config, sizeof(InferenceConfig));
    
    // Initialize KV cache
    size_t seq_capacity = config->max_seq_len > 0 ? config->max_seq_len : 2048;
    int result = kvcache_init(&new_ctx->kv_cache, model, seq_capacity);
    if (result != ONEBIT_SUCCESS) {
        free(new_ctx);
        return result;
    }
    
    // Allocate intermediate buffers
    size_t hidden_size = model_get_hidden_size(model);
    size_t vocab_size = model_get_vocab_size(model);
    
    new_ctx->attention_buffer = malloc(hidden_size * sizeof(float));
    new_ctx->ffn_buffer = malloc(hidden_size * sizeof(float));
    new_ctx->logits_buffer = malloc(vocab_size * sizeof(float));
    
    if (!new_ctx->attention_buffer || !new_ctx->ffn_buffer || !new_ctx->logits_buffer) {
        inference_cleanup(new_ctx);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Initialize online learning buffers if needed
    if (config->enable_online_learning) {
        new_ctx->online_learning_enabled = true;
        new_ctx->inference_count = 0;
        new_ctx->input_size = model_get_input_size(model);
        new_ctx->output_size = model_get_output_size(model);
        
        new_ctx->previous_input = malloc(new_ctx->input_size * sizeof(float));
        new_ctx->expected_output = malloc(new_ctx->output_size * sizeof(float));
        
        if (!new_ctx->previous_input || !new_ctx->expected_output) {
            inference_cleanup(new_ctx);
            return ONEBIT_ERROR_MEMORY;
        }
        
        // Clear buffers
        memset(new_ctx->previous_input, 0, new_ctx->input_size * sizeof(float));
        memset(new_ctx->expected_output, 0, new_ctx->output_size * sizeof(float));
    }
    
    new_ctx->initialized = true;
    *ctx = new_ctx;
    
    return ONEBIT_SUCCESS;
}

void inference_cleanup(InferenceContext* ctx) {
    if (!ctx) return;
    
    // Free KV cache
    kvcache_free(&ctx->kv_cache);
    
    // Free buffers
    free(ctx->attention_buffer);
    free(ctx->ffn_buffer);
    free(ctx->logits_buffer);
    
    // Free online learning buffers
    if (ctx->online_learning_enabled) {
        free(ctx->previous_input);
        free(ctx->expected_output);
    }
    
    free(ctx);
}

int inference_generate(InferenceContext* ctx, const char* prompt,
                      char* output, size_t* output_size,
                      size_t max_tokens) {
    if (!ctx || !prompt || !output || !output_size) return ONEBIT_ERROR_INVALID_PARAM;
    
    // Tokenize prompt
    Token* tokens = malloc(ctx->config.max_sequence_length * sizeof(Token));
    if (!tokens) return ONEBIT_ERROR_MEMORY;
    
    size_t num_tokens = ctx->config.max_sequence_length;
    int result = tokenizer_encode(ctx->tokenizer, prompt, tokens, &num_tokens);
    
    if (result != ONEBIT_SUCCESS) {
        free(tokens);
        return result;
    }
    
    // Generate tokens
    size_t generated = 0;
    uint32_t next_token;
    
    while (generated < max_tokens) {
        result = inference_next_token(ctx, (uint32_t*)tokens, num_tokens + generated,
                                    &next_token);
        
        if (result != ONEBIT_SUCCESS) {
            free(tokens);
            return result;
        }
        
        tokens[num_tokens + generated].id = next_token;
        generated++;
        
        // Check for end of sequence token
        if (next_token == model_get_eos_token(ctx->model)) break;
    }
    
    // Decode generated tokens
    result = tokenizer_decode(ctx->tokenizer, tokens, num_tokens + generated,
                            output, output_size);
    
    free(tokens);
    return result;
}

int inference_next_token(InferenceContext* ctx, const uint32_t* input_tokens,
                        size_t num_tokens, uint32_t* next_token) {
    if (!ctx || !ctx->initialized || !input_tokens || !next_token) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Check if sequence is too long
    if (ctx->current_seq_len + num_tokens > ctx->kv_cache.capacity) {
        return ONEBIT_ERROR_SEQUENCE_TOO_LONG;
    }
    
    // Get model vocab size
    size_t vocab_size = model_get_vocab_size(ctx->model);
    
    // Run forward pass to get logits
    int result = model_forward_with_cache(ctx->model, input_tokens, num_tokens,
                                       ctx->logits_buffer, &ctx->kv_cache, 
                                       ctx->current_seq_len);
    if (result != ONEBIT_SUCCESS) {
        return result;
    }
    
    // Sample next token
    *next_token = sample_token(ctx->logits_buffer, vocab_size,
                             ctx->config.temperature, 
                             ctx->config.top_p,
                             ctx->config.top_k);
    
    // Update sequence length
    ctx->current_seq_len += num_tokens;
    
    // Online learning integration
    if (ctx->online_learning_enabled && ctx->config.online_learning_config.update_during_inference) {
        ctx->inference_count++;
        
        // Store current input for potential learning updates
        if (num_tokens > 0 && ctx->input_size >= num_tokens) {
            // This is a simplified example - real implementation would properly 
            // encode tokens into model input format
            for (size_t i = 0; i < num_tokens && i < ctx->input_size; i++) {
                ctx->previous_input[i] = (float)input_tokens[i];
            }
        }
        
        // Check if we have an expected output from previous interaction
        // and if it's time to perform an update based on frequency
        if (ctx->inference_count > 1 && 
            ctx->inference_count % ctx->config.online_learning_config.update_frequency == 0) {
            
            // Perform online update
            float update_loss = 0.0f;
            onebit_online_update(ctx->model->context, 
                              ctx->previous_input, 
                              ctx->expected_output, 
                              &update_loss);
        }
        
        // Store current prediction as expected output for next feedback loop
        // Real implementation would need more sophisticated feedback mechanism
        if (ctx->output_size > 0) {
            // In a real system, would use the actual logits rather than just the token
            memset(ctx->expected_output, 0, ctx->output_size * sizeof(float));
            ctx->expected_output[*next_token % ctx->output_size] = 1.0f;  // One-hot encoding
        }
    }
    
    return ONEBIT_SUCCESS;
}

int inference_get_logits(InferenceContext* ctx, const uint32_t* input_tokens,
                        size_t num_tokens, float* logits, size_t vocab_size) {
    if (!ctx || !input_tokens || !logits) return ONEBIT_ERROR_INVALID_PARAM;
    if (vocab_size != ctx->model->config.vocab_size) return ONEBIT_ERROR_VOCAB_SIZE;
    
    const ModelConfig* config = &ctx->model->config;
    size_t hidden_size = config->hidden_size;
    
    // Process through transformer layers
    for (size_t layer = 0; layer < config->num_layers; layer++) {
        // Self attention
        if (ctx->config.use_cache) {
            // Use KV cache for attention computation
            size_t cache_offset = layer * hidden_size * ctx->config.max_sequence_length;
            cache_read(&ctx->kv_cache, ctx->attention_buffer,
                      (void*)(uintptr_t)cache_offset,
                      hidden_size * num_tokens * sizeof(float));
        }
        
        if (ctx->config.use_flash_attn) {
            // Use flash attention implementation
            // ... (flash attention computation)
        } else {
            // Standard attention computation
            // ... (standard attention computation)
        }
        
        // Feed-forward network
        // ... (FFN computation)
    }
    
    // Final layer norm and projection to vocab
    // ... (final computations)
    
    return ONEBIT_SUCCESS;
}

int inference_set_params(InferenceContext* ctx, const InferenceConfig* config) {
    if (!ctx || !config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Update configuration
    memcpy(&ctx->config, config, sizeof(InferenceConfig));
    
    // Update online learning state if needed
    if (config->enable_online_learning && !ctx->online_learning_enabled) {
        // Enable online learning that wasn't enabled before
        ctx->online_learning_enabled = true;
        ctx->inference_count = 0;
        
        if (!ctx->previous_input) {
            ctx->input_size = model_get_input_size(ctx->model);
            ctx->previous_input = malloc(ctx->input_size * sizeof(float));
            if (!ctx->previous_input) {
                return ONEBIT_ERROR_MEMORY;
            }
            memset(ctx->previous_input, 0, ctx->input_size * sizeof(float));
        }
        
        if (!ctx->expected_output) {
            ctx->output_size = model_get_output_size(ctx->model);
            ctx->expected_output = malloc(ctx->output_size * sizeof(float));
            if (!ctx->expected_output) {
                free(ctx->previous_input);
                ctx->previous_input = NULL;
                return ONEBIT_ERROR_MEMORY;
            }
            memset(ctx->expected_output, 0, ctx->output_size * sizeof(float));
        }
        
        // Initialize online learning in the core system
        onebit_enable_online_learning(ctx->model->context, &config->online_learning_config);
    }
    else if (!config->enable_online_learning && ctx->online_learning_enabled) {
        // Disable online learning
        ctx->online_learning_enabled = false;
        // Keep the buffers allocated as they may be needed again
    }
    
    return ONEBIT_SUCCESS;
}

void inference_reset(InferenceContext* ctx) {
    if (!ctx) return;
    
    // Reset sequence length and KV cache
    ctx->current_seq_len = 0;
    kvcache_clear(&ctx->kv_cache);
    
    // Reset online learning state
    if (ctx->online_learning_enabled) {
        ctx->inference_count = 0;
        memset(ctx->previous_input, 0, ctx->input_size * sizeof(float));
        memset(ctx->expected_output, 0, ctx->output_size * sizeof(float));
        
        // Reset the online learning state in the core system
        onebit_reset_online_learning(ctx->model->context);
    }
}

bool inference_is_initialized(const InferenceContext* ctx) {
    return ctx ? ctx->initialized : false;
}

const char* inference_get_error(const InferenceContext* ctx) {
    return ctx ? ctx->error_msg : "Invalid context";
}

// New function to provide feedback for online learning
int inference_provide_feedback(InferenceContext* ctx, const float* target_output, float* loss) {
    if (!ctx || !target_output) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    if (!ctx->online_learning_enabled) {
        return ONEBIT_ERROR_NOT_ENABLED;
    }
    
    // Use the stored input and the provided target to update the model
    return onebit_online_update(ctx->model->context, 
                             ctx->previous_input, 
                             target_output, 
                             loss);
}

// ... (remaining implementation of other functions) 