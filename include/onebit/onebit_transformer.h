/**
 * @file onebit_transformer.h
 * @brief Transformer architecture implementation with binary optimization
 */

#ifndef ONEBIT_onebit_transformer_H
#define ONEBIT_onebit_transformer_H

#include "onebit_ops.h"
#include "onebit_data.h"
#include <stdint.h>
#include <stdbool.h>

// Forward declaration
typedef struct OneBitContext OneBitContext;

// Attention configuration
typedef struct {
    size_t hidden_size;      // Hidden dimension size
    size_t num_heads;        // Number of attention heads
    float dropout_prob;      // Attention dropout probability
    bool use_binary;         // Whether to use binary weights
    bool causal_mask;        // Whether to apply causal attention mask
} AttentionConfig;

// Attention layer
typedef struct {
    // Weights
    union {
        struct {
            float* query_weights;    // Query projection weights
            float* key_weights;      // Key projection weights
            float* value_weights;    // Value projection weights
            float* output_weights;   // Output projection weights
        } float_weights;
        
        struct {
            BinaryTensor query_weights;   // Binary query projection
            BinaryTensor key_weights;     // Binary key projection
            BinaryTensor value_weights;   // Binary value projection
            BinaryTensor output_weights;  // Binary output projection
        } binary_weights;
    } weights;
    
    // Biases
    float* query_bias;       // Query projection bias
    float* key_bias;         // Key projection bias
    float* value_bias;       // Value projection bias
    float* output_bias;      // Output projection bias
    
    // Configuration
    AttentionConfig config;
    bool is_binary;          // Whether using binary weights
} AttentionLayer;

// Feed-forward network configuration
typedef struct {
    size_t hidden_size;      // Hidden dimension size
    size_t intermediate_size; // Intermediate dimension size
    float dropout_prob;      // Feed-forward dropout probability
    bool use_binary;         // Whether to use binary weights
} FFNConfig;

// Feed-forward network
typedef struct {
    // Weights
    union {
        struct {
            float* intermediate_weights; // First projection weights
            float* output_weights;       // Second projection weights
        } float_weights;
        
        struct {
            BinaryTensor intermediate_weights; // Binary first projection
            BinaryTensor output_weights;       // Binary second projection
        } binary_weights;
    } weights;
    
    // Biases
    float* intermediate_bias; // First projection bias
    float* output_bias;       // Second projection bias
    
    // Configuration
    FFNConfig config;
    bool is_binary;           // Whether using binary weights
} FFNLayer;

// Transformer block
typedef struct {
    AttentionLayer attention;   // Multi-head attention layer
    FFNLayer ffn;               // Feed-forward network
    
    // Layer normalization parameters
    float* attention_ln_weight; // Attention layer norm weight
    float* attention_ln_bias;   // Attention layer norm bias
    float* ffn_ln_weight;       // FFN layer norm weight
    float* ffn_ln_bias;         // FFN layer norm bias
    
    // Configuration
    bool is_binary;             // Whether using binary weights
    float dropout_prob;         // Residual dropout probability
} TransformerBlock;

// Transformer model
typedef struct {
    // Embedding layers
    float* token_embeddings;    // Token embedding table
    float* position_embeddings; // Position embedding table
    
    // Transformer blocks
    TransformerBlock* blocks;   // Array of transformer blocks
    size_t num_blocks;          // Number of transformer blocks
    
    // Output layer
    union {
        float* output_weights;       // Output projection weights
        BinaryTensor output_weights; // Binary output projection
    } output;
    float* output_bias;         // Output projection bias
    
    // Layer normalization
    float* final_ln_weight;     // Final layer norm weight
    float* final_ln_bias;       // Final layer norm bias
    
    // Configuration
    size_t hidden_size;         // Hidden dimension size
    size_t vocab_size;          // Vocabulary size
    size_t max_seq_len;         // Maximum sequence length
    bool is_binary;             // Whether using binary weights
} TransformerModel;

/**
 * @brief Initialize an attention layer
 * 
 * @param ctx Context for memory allocation
 * @param layer Attention layer to initialize
 * @param config Configuration for the attention layer
 * @return int Error code
 */
int attention_init(OneBitContext* ctx, AttentionLayer* layer, const AttentionConfig* config);

/**
 * @brief Run forward pass through attention layer
 * 
 * @param ctx Execution context
 * @param layer Attention layer
 * @param input Input tensor [batch_size, seq_len, hidden_size]
 * @param output Output tensor [batch_size, seq_len, hidden_size]
 * @param mask Attention mask (optional, can be NULL)
 * @param kv_cache Key-value cache for inference (optional, can be NULL)
 * @return int Error code
 */
int attention_forward(OneBitContext* ctx, AttentionLayer* layer, 
                     const Tensor* input, Tensor* output,
                     const Tensor* mask, void* kv_cache);

/**
 * @brief Initialize a feed-forward layer
 * 
 * @param ctx Context for memory allocation
 * @param layer FFN layer to initialize
 * @param config Configuration for the FFN layer
 * @return int Error code
 */
int ffn_init(OneBitContext* ctx, FFNLayer* layer, const FFNConfig* config);

/**
 * @brief Run forward pass through FFN layer
 * 
 * @param ctx Execution context
 * @param layer FFN layer
 * @param input Input tensor [batch_size, seq_len, hidden_size]
 * @param output Output tensor [batch_size, seq_len, hidden_size]
 * @return int Error code
 */
int ffn_forward(OneBitContext* ctx, FFNLayer* layer,
               const Tensor* input, Tensor* output);

/**
 * @brief Initialize a transformer block
 * 
 * @param ctx Context for memory allocation
 * @param block Transformer block to initialize
 * @param hidden_size Hidden dimension size
 * @param num_heads Number of attention heads
 * @param ff_size Feed-forward intermediate size
 * @param use_binary Whether to use binary weights
 * @return int Error code
 */
int transformer_block_init(OneBitContext* ctx, TransformerBlock* block,
                         size_t hidden_size, size_t num_heads,
                         size_t ff_size, bool use_binary);

/**
 * @brief Run forward pass through transformer block
 * 
 * @param ctx Execution context
 * @param block Transformer block
 * @param input Input tensor [batch_size, seq_len, hidden_size]
 * @param output Output tensor [batch_size, seq_len, hidden_size]
 * @param mask Attention mask (optional, can be NULL)
 * @param kv_cache Key-value cache for inference (optional, can be NULL)
 * @return int Error code
 */
int transformer_block_forward(OneBitContext* ctx, TransformerBlock* block,
                            const Tensor* input, Tensor* output,
                            const Tensor* mask, void* kv_cache);

/**
 * @brief Initialize a complete transformer model
 * 
 * @param ctx Context for memory allocation
 * @param model Transformer model to initialize
 * @param hidden_size Hidden dimension size
 * @param num_layers Number of transformer layers
 * @param num_heads Number of attention heads
 * @param ff_size Feed-forward intermediate size
 * @param vocab_size Vocabulary size
 * @param max_seq_len Maximum sequence length
 * @param use_binary Whether to use binary weights
 * @return int Error code
 */
int transformer_model_init(OneBitContext* ctx, TransformerModel* model,
                         size_t hidden_size, size_t num_layers,
                         size_t num_heads, size_t ff_size,
                         size_t vocab_size, size_t max_seq_len,
                         bool use_binary);

/**
 * @brief Run forward pass through transformer model
 * 
 * @param ctx Execution context
 * @param model Transformer model
 * @param input_ids Input token IDs [batch_size, seq_len]
 * @param output Output logits [batch_size, seq_len, vocab_size]
 * @param kv_cache Key-value cache for inference (optional, can be NULL)
 * @return int Error code
 */
int transformer_model_forward(OneBitContext* ctx, TransformerModel* model,
                            const int64_t* input_ids, Tensor* output,
                            void* kv_cache);

/**
 * @brief Cleanup and free a transformer model
 * 
 * @param ctx Context for memory deallocation
 * @param model Transformer model to cleanup
 */
void transformer_model_cleanup(OneBitContext* ctx, TransformerModel* model);

#endif // ONEBIT_TRANSFORMER_H 