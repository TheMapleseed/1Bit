/**
 * @file onebit_data.h
 * @brief Data handling structures and functions for OneBit framework
 * @author OneBit Team
 * @version 0.1.0
 * @date 2023-03-17
 */

#ifndef ONEBIT_onebit_data_H
#define ONEBIT_onebit_data_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Data types supported by the framework
 */
typedef enum {
    DATA_TYPE_FLOAT,       /**< 32-bit floating point */
    DATA_TYPE_FLOAT16,     /**< 16-bit floating point */
    DATA_TYPE_BFLOAT16,    /**< 16-bit brain floating point */
    DATA_TYPE_INT8,        /**< 8-bit integer */
    DATA_TYPE_UINT8,       /**< 8-bit unsigned integer */
    DATA_TYPE_INT4,        /**< 4-bit integer */
    DATA_TYPE_BINARY,      /**< Binary (1-bit) */
} DataType;

/**
 * @brief Data layout types
 */
typedef enum {
    DATA_LAYOUT_ROW_MAJOR,     /**< Row-major data layout */
    DATA_LAYOUT_COLUMN_MAJOR,  /**< Column-major data layout */
    DATA_LAYOUT_PACKED_BINARY, /**< Packed binary representation */
} DataLayout;

/**
 * @brief Tokenized sequence structure
 */
typedef struct {
    uint32_t* tokens;           /**< Array of token IDs */
    size_t length;              /**< Length of the token sequence */
    size_t capacity;            /**< Capacity of the tokens array */
} TokenSequence;

/**
 * @brief Tensor structure that represents n-dimensional data
 */
typedef struct {
    void* data;                 /**< Pointer to the raw data */
    size_t* shape;              /**< Array of dimension sizes */
    size_t* strides;            /**< Array of strides for each dimension */
    size_t ndim;                /**< Number of dimensions */
    size_t size;                /**< Total number of elements */
    size_t bytes;               /**< Total size in bytes */
    DataType dtype;             /**< Data type */
    DataLayout layout;          /**< Data layout */
    bool owns_data;             /**< Whether this tensor owns its data */
    int device_id;              /**< Device ID where the data resides */
    int flags;                  /**< Additional flags */
} Tensor;

/**
 * @brief Dataset sample structure
 */
typedef struct {
    Tensor* input;              /**< Input tensor */
    Tensor* output;             /**< Output tensor (may be NULL for inference) */
    const char* id;             /**< Sample identifier */
    void* metadata;             /**< Optional metadata */
} DataSample;

/**
 * @brief Batch of data samples
 */
typedef struct {
    Tensor* input_batch;        /**< Batched input tensor */
    Tensor* output_batch;       /**< Batched output tensor */
    size_t batch_size;          /**< Number of samples in the batch */
    size_t max_seq_len;         /**< Maximum sequence length in this batch */
    void* metadata;             /**< Optional batch metadata */
} DataBatch;

/**
 * @brief Dataset iterator configuration
 */
typedef struct {
    size_t batch_size;          /**< Batch size */
    bool shuffle;               /**< Whether to shuffle the data */
    size_t seed;                /**< Random seed for shuffling */
    size_t max_seq_len;         /**< Maximum sequence length (for padding/truncation) */
    bool drop_last;             /**< Whether to drop the last incomplete batch */
    size_t prefetch_size;       /**< Number of batches to prefetch */
    size_t num_workers;         /**< Number of worker threads for data loading */
} DataIterConfig;

/**
 * @brief Dataset structure
 */
typedef struct {
    const char* name;           /**< Dataset name */
    size_t size;                /**< Number of samples */
    void* data;                 /**< Raw dataset data */
    bool owns_data;             /**< Whether this dataset owns its data */
    void* metadata;             /**< Dataset metadata */
    void* iter_state;           /**< Iterator state (private) */
    DataIterConfig iter_config; /**< Iterator configuration */
} Dataset;

/**
 * @brief Vocabulary for tokenization
 */
typedef struct {
    const char** tokens;         /**< Array of token strings */
    size_t size;                /**< Number of tokens in vocabulary */
    const char* unk_token;      /**< Unknown token string */
    const char* pad_token;      /**< Padding token string */
    const char* bos_token;      /**< Beginning of sequence token string */
    const char* eos_token;      /**< End of sequence token string */
    uint32_t unk_token_id;      /**< Unknown token ID */
    uint32_t pad_token_id;      /**< Padding token ID */
    uint32_t bos_token_id;      /**< Beginning of sequence token ID */
    uint32_t eos_token_id;      /**< End of sequence token ID */
    void* encoding_trie;        /**< Encoding trie for fast lookup (private) */
} Vocabulary;

/**
 * @brief Tokenizer for text processing
 */
typedef struct {
    Vocabulary* vocab;          /**< Vocabulary */
    bool add_bos_token;         /**< Whether to add beginning token */
    bool add_eos_token;         /**< Whether to add end token */
    void* tokenizer_state;      /**< Tokenizer specific state (private) */
} Tokenizer;

// === Tensor functions ===

/**
 * @brief Initialize a tensor with given properties
 * @param tensor Pointer to the tensor to initialize
 * @param data Data pointer (if NULL, memory will be allocated)
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param dtype Data type
 * @param layout Data layout
 * @param flags Additional flags
 * @return Error code (0 for success)
 */
int tensor_init(Tensor* tensor, void* data, const size_t* shape, size_t ndim, 
               DataType dtype, DataLayout layout, int flags);

/**
 * @brief Clean up a tensor and free its resources
 * @param tensor Tensor to clean up
 */
void tensor_cleanup(Tensor* tensor);

/**
 * @brief Create a tensor on a specific device
 * @param tensor Pointer to store the created tensor
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param dtype Data type
 * @param device_id Device ID (-1 for CPU)
 * @return Error code (0 for success)
 */
int tensor_create(Tensor** tensor, const size_t* shape, size_t ndim, 
                 DataType dtype, int device_id);

/**
 * @brief Copy a tensor to another device or memory space
 * @param src Source tensor
 * @param dest Destination tensor (can be NULL to create a new tensor)
 * @param device_id Target device ID (-1 for CPU)
 * @param flags Memory flags
 * @return Pointer to the destination tensor
 */
Tensor* tensor_to_device(const Tensor* src, Tensor* dest, int device_id, int flags);

/**
 * @brief Reshape a tensor without changing its data
 * @param tensor Tensor to reshape
 * @param new_shape New shape dimensions
 * @param new_ndim Number of dimensions in new shape
 * @return Error code (0 for success)
 */
int tensor_reshape(Tensor* tensor, const size_t* new_shape, size_t new_ndim);

/**
 * @brief Get the size in bytes for a given data type
 * @param dtype Data type
 * @return Size in bytes
 */
size_t get_dtype_size(DataType dtype);

/**
 * @brief Get the element at the specified indices
 * @param tensor Source tensor
 * @param indices Array of indices for each dimension
 * @param value Pointer to store the retrieved value (type depends on tensor dtype)
 * @return Error code (0 for success)
 */
int tensor_get(const Tensor* tensor, const size_t* indices, void* value);

/**
 * @brief Set the element at the specified indices
 * @param tensor Target tensor
 * @param indices Array of indices for each dimension
 * @param value Pointer to the value to set (type depends on tensor dtype)
 * @return Error code (0 for success)
 */
int tensor_set(Tensor* tensor, const size_t* indices, const void* value);

// === Dataset functions ===

/**
 * @brief Initialize a dataset with the given configuration
 * @param dataset Pointer to the dataset to initialize
 * @param name Dataset name
 * @param size Number of samples
 * @param config Iterator configuration
 * @return Error code (0 for success)
 */
int dataset_init(Dataset* dataset, const char* name, size_t size, 
                const DataIterConfig* config);

/**
 * @brief Clean up a dataset and free its resources
 * @param dataset Dataset to clean up
 */
void dataset_cleanup(Dataset* dataset);

/**
 * @brief Get the next batch from the dataset
 * @param dataset Source dataset
 * @param batch Pointer to store the batch
 * @return Error code (0 for success, ONEBIT_ERROR_END_OF_DATA when no more batches)
 */
int dataset_next_batch(Dataset* dataset, DataBatch* batch);

/**
 * @brief Reset the dataset iterator
 * @param dataset Dataset to reset
 * @param shuffle Whether to shuffle the data
 * @return Error code (0 for success)
 */
int dataset_reset(Dataset* dataset, bool shuffle);

/**
 * @brief Load a dataset from a file or directory
 * @param path Path to the dataset file or directory
 * @param config Iterator configuration
 * @param dataset Pointer to store the loaded dataset
 * @return Error code (0 for success)
 */
int dataset_load(const char* path, const DataIterConfig* config, Dataset** dataset);

/**
 * @brief Split a dataset into training and validation sets
 * @param dataset Source dataset
 * @param train_ratio Ratio of samples to use for training (0.0-1.0)
 * @param train_dataset Pointer to store the training dataset
 * @param val_dataset Pointer to store the validation dataset
 * @param seed Random seed for shuffling
 * @return Error code (0 for success)
 */
int dataset_split(const Dataset* dataset, float train_ratio,
                 Dataset** train_dataset, Dataset** val_dataset, size_t seed);

// === Tokenization functions ===

/**
 * @brief Initialize a tokenizer with the given vocabulary
 * @param tokenizer Pointer to the tokenizer to initialize
 * @param vocab Vocabulary to use
 * @return Error code (0 for success)
 */
int tokenizer_init(Tokenizer* tokenizer, Vocabulary* vocab);

/**
 * @brief Clean up a tokenizer and free its resources
 * @param tokenizer Tokenizer to clean up
 */
void tokenizer_cleanup(Tokenizer* tokenizer);

/**
 * @brief Load a vocabulary from a file
 * @param path Path to the vocabulary file
 * @param vocab Pointer to store the loaded vocabulary
 * @return Error code (0 for success)
 */
int vocab_load(const char* path, Vocabulary** vocab);

/**
 * @brief Create a new vocabulary from a token list
 * @param tokens Array of token strings
 * @param size Number of tokens
 * @param vocab Pointer to store the created vocabulary
 * @return Error code (0 for success)
 */
int vocab_create(const char** tokens, size_t size, Vocabulary** vocab);

/**
 * @brief Clean up a vocabulary and free its resources
 * @param vocab Vocabulary to clean up
 */
void vocab_cleanup(Vocabulary* vocab);

/**
 * @brief Encode a text string into token IDs
 * @param tokenizer Tokenizer to use
 * @param text Input text string
 * @param sequence Pointer to store the tokenized sequence
 * @return Error code (0 for success)
 */
int tokenizer_encode(const Tokenizer* tokenizer, const char* text, TokenSequence* sequence);

/**
 * @brief Decode token IDs back into text
 * @param tokenizer Tokenizer to use
 * @param sequence Token sequence to decode
 * @param text Pointer to store the decoded text (must be freed by caller)
 * @return Error code (0 for success)
 */
int tokenizer_decode(const Tokenizer* tokenizer, const TokenSequence* sequence, char** text);

/**
 * @brief Initialize a token sequence
 * @param sequence Pointer to the sequence to initialize
 * @param capacity Initial capacity for tokens
 * @return Error code (0 for success)
 */
int token_sequence_init(TokenSequence* sequence, size_t capacity);

/**
 * @brief Clean up a token sequence and free its resources
 * @param sequence Sequence to clean up
 */
void token_sequence_cleanup(TokenSequence* sequence);

/**
 * @brief Add a token to a sequence
 * @param sequence Target sequence
 * @param token_id Token ID to add
 * @return Error code (0 for success)
 */
int token_sequence_add(TokenSequence* sequence, uint32_t token_id);

#ifdef __cplusplus
}
#endif

#endif /* ONEBIT_DATA_H */ 