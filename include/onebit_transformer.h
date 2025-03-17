/**
 * @file onebit_transformer.h
 * @brief 1-bit Transformer model implementation
 */

#ifndef ONEBIT_TRANSFORMER_H
#define ONEBIT_TRANSFORMER_H

#include "onebit_context.h"
#include "onebit_tensor.h"
#include "onebit_attention.h"
#include "onebit_ffn.h"
#include "onebit_kv_cache.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Transformer block structure
 */
typedef struct {
    /** Attention layer */
    AttentionLayer attention;
    
    /** Feed-forward layer */
    FFNLayer ffn;
    
    /** Layer normalization weights for attention */
    float* attention_ln_weight;
    
    /** Layer normalization biases for attention */
    float* attention_ln_bias;
    
    /** Layer normalization weights for FFN */
    float* ffn_ln_weight;
    
    /** Layer normalization biases for FFN */
    float* ffn_ln_bias;
    
    /** Whether to use binary weights */
    bool is_binary;
    
    /** Dropout probability */
    float dropout_prob;
} TransformerBlock;

/**
 * @brief Transformer model structure
 */
typedef struct {
    /** Array of transformer blocks */
    TransformerBlock* blocks;
    
    /** Number of transformer blocks */
    size_t num_blocks;
    
    /** Hidden size */
    size_t hidden_size;
    
    /** Token embeddings */
    float* token_embeddings;
    
    /** Position embeddings */
    float* position_embeddings;
    
    /** Vocabulary size */
    size_t vocab_size;
    
    /** Maximum sequence length */
    size_t max_seq_len;
    
    /** Whether to use binary weights */
    bool is_binary;
    
    /** Output projection layer */
    union {
        float* output_weights;  /** < Dense weights */
        BinaryTensor output_weights;  /** < Binary weights */
    } output;
    
    /** Output projection bias */
    float* output_bias;
    
    /** Final layer normalization weights */
    float* final_ln_weight;
    
    /** Final layer normalization biases */
    float* final_ln_bias;
} TransformerModel;

/**
 * @brief Initialize a transformer block
 * 
 * @param ctx Context for memory management
 * @param block Pointer to transformer block to initialize
 * @param hidden_size Size of hidden dimension
 * @param num_heads Number of attention heads
 * @param ff_size Size of feed-forward intermediate layer
 * @param use_binary Whether to use binary weights
 * @return ONEBIT_SUCCESS on success, error code otherwise
 */
int transformer_block_init(OneBitContext* ctx, TransformerBlock* block,
                         size_t hidden_size, size_t num_heads,
                         size_t ff_size, bool use_binary);

/**
 * @brief Clean up a transformer block
 * 
 * @param ctx Context for memory management
 * @param block Pointer to transformer block to clean up
 */
void transformer_block_cleanup(OneBitContext* ctx, TransformerBlock* block);

/**
 * @brief Forward pass through transformer block
 * 
 * @param ctx Context for memory management
 * @param block Pointer to transformer block
 * @param input Input tensor (shape: [batch_size, seq_len, hidden_size])
 * @param output Output tensor (shape: [batch_size, seq_len, hidden_size])
 * @param mask Optional attention mask tensor
 * @param kv_cache Optional key-value cache for incremental decoding
 * @return ONEBIT_SUCCESS on success, error code otherwise
 */
int transformer_block_forward(OneBitContext* ctx, TransformerBlock* block,
                            const Tensor* input, Tensor* output,
                            const Tensor* mask, void* kv_cache);

/**
 * @brief Initialize a transformer model
 * 
 * @param ctx Context for memory management
 * @param model Pointer to transformer model to initialize
 * @param hidden_size Size of hidden dimension
 * @param num_layers Number of transformer layers
 * @param num_heads Number of attention heads
 * @param ff_size Size of feed-forward intermediate layer
 * @param vocab_size Size of vocabulary
 * @param max_seq_len Maximum sequence length
 * @param use_binary Whether to use binary weights
 * @return ONEBIT_SUCCESS on success, error code otherwise
 */
int transformer_model_init(OneBitContext* ctx, TransformerModel* model,
                         size_t hidden_size, size_t num_layers,
                         size_t num_heads, size_t ff_size,
                         size_t vocab_size, size_t max_seq_len,
                         bool use_binary);

/**
 * @brief Clean up a transformer model
 * 
 * @param ctx Context for memory management
 * @param model Pointer to transformer model to clean up
 */
void transformer_model_cleanup(OneBitContext* ctx, TransformerModel* model);

/**
 * @brief Forward pass through transformer model
 * 
 * @param ctx Context for memory management
 * @param model Pointer to transformer model
 * @param input_ids Input token IDs (shape: [batch_size, seq_len])
 * @param output Output tensor (shape: [batch_size, seq_len, vocab_size])
 * @param kv_cache_ptr Optional key-value cache for incremental decoding
 * @return ONEBIT_SUCCESS on success, error code otherwise
 */
int transformer_model_forward(OneBitContext* ctx, TransformerModel* model,
                            const int64_t* input_ids, Tensor* output,
                            void* kv_cache_ptr);

/**
 * @brief Create KV cache for efficient inference
 * 
 * @param ctx Context for memory management
 * @param model Pointer to transformer model
 * @param batch_size Batch size for inference
 * @param kv_cache Pointer to store KV cache
 * @return ONEBIT_SUCCESS on success, error code otherwise
 */
int transformer_create_kv_cache(OneBitContext* ctx, TransformerModel* model,
                              size_t batch_size, void** kv_cache);

/**
 * @brief Free KV cache
 * 
 * @param ctx Context for memory management
 * @param model Pointer to transformer model
 * @param kv_cache Pointer to KV cache to free
 */
void transformer_free_kv_cache(OneBitContext* ctx, TransformerModel* model, 
                             void* kv_cache);

#ifdef __cplusplus
}
#endif

#endif /* ONEBIT_TRANSFORMER_H */ 