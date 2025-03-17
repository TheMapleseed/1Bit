#include <onebit/onebit_data.h>
#include <onebit/onebit_error.h>
#include <onebit/onebit_memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <math.h>

int data_init(DataContext* ctx, const DataConfig* config) {
    if (!ctx || !config || config->element_size == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->data = malloc(config->initial_capacity * config->element_size);
    if (!ctx->data) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    ctx->size = 0;
    ctx->capacity = config->initial_capacity;
    ctx->element_size = config->element_size;
    ctx->growth_factor = config->growth_factor > 1.0f ?
                        config->growth_factor : 2.0f;
    
    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        free(ctx->data);
        return ONEBIT_ERROR_THREAD;
    }
    
    return ONEBIT_SUCCESS;
}

void data_cleanup(DataContext* ctx) {
    if (!ctx) return;
    
    free(ctx->data);
    pthread_mutex_destroy(&ctx->mutex);
}

static int data_grow(DataContext* ctx, size_t min_capacity) {
    size_t new_capacity = ctx->capacity;
    
    while (new_capacity < min_capacity) {
        new_capacity = (size_t)(new_capacity * ctx->growth_factor);
    }
    
    void* new_data = realloc(ctx->data,
                            new_capacity * ctx->element_size);
    if (!new_data) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    ctx->data = new_data;
    ctx->capacity = new_capacity;
    return ONEBIT_SUCCESS;
}

int data_add(DataContext* ctx, const void* data, size_t size) {
    if (!ctx || !data || size == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    if (ctx->size + size > ctx->capacity) {
        int result = data_grow(ctx, ctx->size + size);
        if (result != ONEBIT_SUCCESS) {
            pthread_mutex_unlock(&ctx->mutex);
            return result;
        }
    }
    
    memcpy((uint8_t*)ctx->data + ctx->size * ctx->element_size,
           data, size * ctx->element_size);
    ctx->size += size;
    
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
}

int data_get(const DataContext* ctx, size_t index, void* data) {
    if (!ctx || !data || index >= ctx->size) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock((pthread_mutex_t*)&ctx->mutex);
    
    memcpy(data,
           (uint8_t*)ctx->data + index * ctx->element_size,
           ctx->element_size);
    
    pthread_mutex_unlock((pthread_mutex_t*)&ctx->mutex);
    return ONEBIT_SUCCESS;
}

size_t data_size(const DataContext* ctx) {
    return ctx ? ctx->size : 0;
}

size_t data_capacity(const DataContext* ctx) {
    return ctx ? ctx->capacity : 0;
}

void data_clear(DataContext* ctx) {
    if (!ctx) return;
    
    pthread_mutex_lock(&ctx->mutex);
    ctx->size = 0;
    pthread_mutex_unlock(&ctx->mutex);
}

int data_reserve(DataContext* ctx, size_t capacity) {
    if (!ctx) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    if (capacity > ctx->capacity) {
        int result = data_grow(ctx, capacity);
        if (result != ONEBIT_SUCCESS) {
            pthread_mutex_unlock(&ctx->mutex);
            return result;
        }
    }
    
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
}

void* data_get_buffer(const DataContext* ctx) {
    return ctx ? ctx->data : NULL;
}

// Get size in bytes for a data type
size_t get_dtype_size(DataType dtype) {
    switch (dtype) {
        case DATA_TYPE_FLOAT:
            return 4;
        case DATA_TYPE_FLOAT16:
        case DATA_TYPE_BFLOAT16:
            return 2;
        case DATA_TYPE_INT8:
        case DATA_TYPE_UINT8:
            return 1;
        case DATA_TYPE_INT4:
            return 0.5; // Half a byte, must be packed
        case DATA_TYPE_BINARY:
            return 0.125; // 1/8 of a byte, must be packed
        default:
            return 0;
    }
}

// Calculate total size and strides for a tensor
static void calculate_tensor_layout(Tensor* tensor) {
    if (!tensor || !tensor->shape || tensor->ndim == 0) {
        return;
    }
    
    // Calculate total size
    tensor->size = 1;
    for (size_t i = 0; i < tensor->ndim; i++) {
        tensor->size *= tensor->shape[i];
    }
    
    // Calculate strides
    if (!tensor->strides) {
        tensor->strides = (size_t*)malloc(tensor->ndim * sizeof(size_t));
        if (!tensor->strides) {
            return;
        }
    }
    
    if (tensor->layout == DATA_LAYOUT_ROW_MAJOR) {
        // Row-major (C-style) strides
        tensor->strides[tensor->ndim - 1] = 1;
        for (int i = (int)tensor->ndim - 2; i >= 0; i--) {
            tensor->strides[i] = tensor->strides[i + 1] * tensor->shape[i + 1];
        }
    } else if (tensor->layout == DATA_LAYOUT_COLUMN_MAJOR) {
        // Column-major (Fortran-style) strides
        tensor->strides[0] = 1;
        for (size_t i = 1; i < tensor->ndim; i++) {
            tensor->strides[i] = tensor->strides[i - 1] * tensor->shape[i - 1];
        }
    } else if (tensor->layout == DATA_LAYOUT_PACKED_BINARY) {
        // Packed binary layout - special case
        if (tensor->dtype == DATA_TYPE_BINARY) {
            // For binary, we pack 8 elements per byte
            tensor->strides[tensor->ndim - 1] = 0.125; // 1/8 for bit packing
            for (int i = (int)tensor->ndim - 2; i >= 0; i--) {
                tensor->strides[i] = tensor->strides[i + 1] * tensor->shape[i + 1];
            }
        } else if (tensor->dtype == DATA_TYPE_INT4) {
            // For INT4, we pack 2 elements per byte
            tensor->strides[tensor->ndim - 1] = 0.5; // 1/2 for nibble packing
            for (int i = (int)tensor->ndim - 2; i >= 0; i--) {
                tensor->strides[i] = tensor->strides[i + 1] * tensor->shape[i + 1];
            }
        }
    }
    
    // Calculate bytes
    double dtype_size = get_dtype_size(tensor->dtype);
    tensor->bytes = (size_t)ceil(tensor->size * dtype_size);
}

// Initialize a tensor
int tensor_init(Tensor* tensor, void* data, const size_t* shape, size_t ndim, 
               DataType dtype, DataLayout layout, int flags) {
    if (!tensor || !shape || ndim == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Clear the tensor struct
    memset(tensor, 0, sizeof(Tensor));
    
    // Copy the shape
    tensor->shape = (size_t*)malloc(ndim * sizeof(size_t));
    if (!tensor->shape) {
        return ONEBIT_ERROR_MEMORY;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    
    // Set tensor properties
    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->layout = layout;
    tensor->flags = flags;
    tensor->device_id = -1; // Default to CPU
    
    // Calculate strides and size
    calculate_tensor_layout(tensor);
    
    // Allocate or use provided data
    if (data) {
        tensor->data = data;
        tensor->owns_data = false;
    } else {
        tensor->data = malloc(tensor->bytes);
        if (!tensor->data) {
            free(tensor->shape);
            free(tensor->strides);
            return ONEBIT_ERROR_MEMORY;
        }
        tensor->owns_data = true;
    }
    
    return ONEBIT_SUCCESS;
}

// Clean up a tensor
void tensor_cleanup(Tensor* tensor) {
    if (!tensor) {
        return;
    }
    
    // Free allocated memory
    if (tensor->owns_data && tensor->data) {
        free(tensor->data);
    }
    
    if (tensor->shape) {
        free(tensor->shape);
    }
    
    if (tensor->strides) {
        free(tensor->strides);
    }
    
    // Clear the struct
    memset(tensor, 0, sizeof(Tensor));
}

// Create a tensor on a device
int tensor_create(Tensor** tensor, const size_t* shape, size_t ndim, 
                 DataType dtype, int device_id) {
    if (!tensor || !shape || ndim == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Allocate tensor struct
    *tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!*tensor) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Initialize with default layout
    DataLayout layout = (dtype == DATA_TYPE_BINARY || dtype == DATA_TYPE_INT4) ? 
                       DATA_LAYOUT_PACKED_BINARY : DATA_LAYOUT_ROW_MAJOR;
    
    int flags = (device_id >= 0) ? MEMORY_FLAG_DEVICE : MEMORY_FLAG_DEFAULT;
    
    // Initialize tensor
    int result = tensor_init(*tensor, NULL, shape, ndim, dtype, layout, flags);
    if (result != ONEBIT_SUCCESS) {
        free(*tensor);
        *tensor = NULL;
        return result;
    }
    
    (*tensor)->device_id = device_id;
    
    // If device memory is requested, allocate it
    if (device_id >= 0) {
        // Free the CPU memory allocated in tensor_init
        if ((*tensor)->owns_data && (*tensor)->data) {
            free((*tensor)->data);
            (*tensor)->data = NULL;
        }
        
        // Allocate device memory
        (*tensor)->data = memory_direct_allocate((*tensor)->bytes, flags, device_id);
        if (!(*tensor)->data) {
            tensor_cleanup(*tensor);
            free(*tensor);
            *tensor = NULL;
            return ONEBIT_ERROR_MEMORY;
        }
        
        (*tensor)->owns_data = true;
    }
    
    return ONEBIT_SUCCESS;
}

// Copy tensor to device
Tensor* tensor_to_device(const Tensor* src, Tensor* dest, int device_id, int flags) {
    if (!src) {
        return NULL;
    }
    
    bool created_dest = false;
    
    // Create destination tensor if not provided
    if (!dest) {
        dest = (Tensor*)malloc(sizeof(Tensor));
        if (!dest) {
            error_set(ONEBIT_ERROR_MEMORY, "Failed to allocate memory for destination tensor");
            return NULL;
        }
        created_dest = true;
        
        // Initialize with same properties as source
        int result = tensor_init(dest, NULL, src->shape, src->ndim, 
                                src->dtype, src->layout, flags);
        if (result != ONEBIT_SUCCESS) {
            free(dest);
            return NULL;
        }
        
        dest->device_id = device_id;
        
        // Free the CPU memory allocated in tensor_init
        if (dest->owns_data && dest->data) {
            free(dest->data);
            dest->data = NULL;
        }
        
        // Allocate device memory
        int mem_flags = (device_id >= 0) ? MEMORY_FLAG_DEVICE | flags : flags;
        dest->data = memory_direct_allocate(dest->bytes, mem_flags, device_id);
        if (!dest->data) {
            if (created_dest) {
                tensor_cleanup(dest);
                free(dest);
            }
            return NULL;
        }
        
        dest->owns_data = true;
    }
    
    // Copy data
    int src_flags = (src->device_id >= 0) ? MEMORY_FLAG_DEVICE : MEMORY_FLAG_DEFAULT;
    int dst_flags = (dest->device_id >= 0) ? MEMORY_FLAG_DEVICE : MEMORY_FLAG_DEFAULT;
    
    int result = memory_copy(dest->data, src->data, src->bytes, dst_flags, src_flags);
    if (result != ONEBIT_SUCCESS) {
        if (created_dest) {
            tensor_cleanup(dest);
            free(dest);
        }
        return NULL;
    }
    
    return dest;
}

// Reshape a tensor
int tensor_reshape(Tensor* tensor, const size_t* new_shape, size_t new_ndim) {
    if (!tensor || !new_shape || new_ndim == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Calculate total elements in new shape
    size_t new_size = 1;
    for (size_t i = 0; i < new_ndim; i++) {
        new_size *= new_shape[i];
    }
    
    // Verify that total number of elements is the same
    if (new_size != tensor->size) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Reallocate shape array if needed
    if (new_ndim != tensor->ndim) {
        size_t* new_shape_copy = (size_t*)realloc(tensor->shape, new_ndim * sizeof(size_t));
        if (!new_shape_copy) {
            return ONEBIT_ERROR_MEMORY;
        }
        tensor->shape = new_shape_copy;
        
        // Free old strides and allocate new
        if (tensor->strides) {
            free(tensor->strides);
            tensor->strides = NULL;
        }
    }
    
    // Copy new shape
    memcpy(tensor->shape, new_shape, new_ndim * sizeof(size_t));
    tensor->ndim = new_ndim;
    
    // Recalculate strides
    calculate_tensor_layout(tensor);
    
    return ONEBIT_SUCCESS;
}

// Get element at index
int tensor_get(const Tensor* tensor, const size_t* indices, void* value) {
    if (!tensor || !indices || !value || !tensor->data) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Calculate flat index
    size_t flat_index = 0;
    for (size_t i = 0; i < tensor->ndim; i++) {
        flat_index += indices[i] * tensor->strides[i];
    }
    
    // Handle different data types
    switch (tensor->dtype) {
        case DATA_TYPE_FLOAT:
            {
                float* float_data = (float*)tensor->data;
                *(float*)value = float_data[flat_index];
            }
            break;
        case DATA_TYPE_FLOAT16:
        case DATA_TYPE_BFLOAT16:
            {
                uint16_t* half_data = (uint16_t*)tensor->data;
                *(uint16_t*)value = half_data[flat_index];
            }
            break;
        case DATA_TYPE_INT8:
            {
                int8_t* int8_data = (int8_t*)tensor->data;
                *(int8_t*)value = int8_data[flat_index];
            }
            break;
        case DATA_TYPE_UINT8:
            {
                uint8_t* uint8_data = (uint8_t*)tensor->data;
                *(uint8_t*)value = uint8_data[flat_index];
            }
            break;
        case DATA_TYPE_INT4:
            {
                // For INT4, 2 values are packed per byte
                uint8_t* packed_data = (uint8_t*)tensor->data;
                size_t byte_index = flat_index / 2;
                bool is_high_nibble = (flat_index % 2 == 0);
                
                if (is_high_nibble) {
                    *(int8_t*)value = (packed_data[byte_index] >> 4) & 0x0F;
                } else {
                    *(int8_t*)value = packed_data[byte_index] & 0x0F;
                }
            }
            break;
        case DATA_TYPE_BINARY:
            {
                // For binary, 8 values are packed per byte
                uint8_t* packed_data = (uint8_t*)tensor->data;
                size_t byte_index = flat_index / 8;
                size_t bit_index = flat_index % 8;
                
                *(bool*)value = (packed_data[byte_index] >> bit_index) & 0x01;
            }
            break;
        default:
            return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    return ONEBIT_SUCCESS;
}

// Set element at index
int tensor_set(Tensor* tensor, const size_t* indices, const void* value) {
    if (!tensor || !indices || !value || !tensor->data) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Calculate flat index
    size_t flat_index = 0;
    for (size_t i = 0; i < tensor->ndim; i++) {
        flat_index += indices[i] * tensor->strides[i];
    }
    
    // Handle different data types
    switch (tensor->dtype) {
        case DATA_TYPE_FLOAT:
            {
                float* float_data = (float*)tensor->data;
                float_data[flat_index] = *(const float*)value;
            }
            break;
        case DATA_TYPE_FLOAT16:
        case DATA_TYPE_BFLOAT16:
            {
                uint16_t* half_data = (uint16_t*)tensor->data;
                half_data[flat_index] = *(const uint16_t*)value;
            }
            break;
        case DATA_TYPE_INT8:
            {
                int8_t* int8_data = (int8_t*)tensor->data;
                int8_data[flat_index] = *(const int8_t*)value;
            }
            break;
        case DATA_TYPE_UINT8:
            {
                uint8_t* uint8_data = (uint8_t*)tensor->data;
                uint8_data[flat_index] = *(const uint8_t*)value;
            }
            break;
        case DATA_TYPE_INT4:
            {
                // For INT4, 2 values are packed per byte
                uint8_t* packed_data = (uint8_t*)tensor->data;
                size_t byte_index = flat_index / 2;
                bool is_high_nibble = (flat_index % 2 == 0);
                
                int8_t nibble_value = *(const int8_t*)value & 0x0F;
                
                if (is_high_nibble) {
                    packed_data[byte_index] = (packed_data[byte_index] & 0x0F) | (nibble_value << 4);
                } else {
                    packed_data[byte_index] = (packed_data[byte_index] & 0xF0) | nibble_value;
                }
            }
            break;
        case DATA_TYPE_BINARY:
            {
                // For binary, 8 values are packed per byte
                uint8_t* packed_data = (uint8_t*)tensor->data;
                size_t byte_index = flat_index / 8;
                size_t bit_index = flat_index % 8;
                
                bool bit_value = *(const bool*)value;
                
                if (bit_value) {
                    packed_data[byte_index] |= (1 << bit_index);
                } else {
                    packed_data[byte_index] &= ~(1 << bit_index);
                }
            }
            break;
        default:
            return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    return ONEBIT_SUCCESS;
}

// Dataset iterator state
typedef struct {
    size_t* indices;         // Shuffled indices
    size_t current_index;    // Current position in indices array
    size_t num_batches;      // Total number of batches
    pthread_mutex_t mutex;   // Mutex for thread safety
} DatasetIterState;

// Initialize a dataset
int dataset_init(Dataset* dataset, const char* name, size_t size, 
                const DataIterConfig* config) {
    if (!dataset || !name || !config || size == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Clear the dataset struct
    memset(dataset, 0, sizeof(Dataset));
    
    // Copy name and config
    dataset->name = strdup(name);
    if (!dataset->name) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    dataset->size = size;
    memcpy(&dataset->iter_config, config, sizeof(DataIterConfig));
    
    // Initialize iterator state
    DatasetIterState* iter_state = (DatasetIterState*)malloc(sizeof(DatasetIterState));
    if (!iter_state) {
        free((void*)dataset->name);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Calculate number of batches
    size_t batch_size = config->batch_size;
    iter_state->num_batches = size / batch_size;
    if (!config->drop_last && size % batch_size != 0) {
        iter_state->num_batches++;
    }
    
    // Allocate indices array
    iter_state->indices = (size_t*)malloc(size * sizeof(size_t));
    if (!iter_state->indices) {
        free(iter_state);
        free((void*)dataset->name);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Initialize indices
    for (size_t i = 0; i < size; i++) {
        iter_state->indices[i] = i;
    }
    
    // Initialize mutex
    if (pthread_mutex_init(&iter_state->mutex, NULL) != 0) {
        free(iter_state->indices);
        free(iter_state);
        free((void*)dataset->name);
        return ONEBIT_ERROR_THREAD;
    }
    
    iter_state->current_index = 0;
    dataset->iter_state = iter_state;
    
    return ONEBIT_SUCCESS;
}

// Clean up a dataset
void dataset_cleanup(Dataset* dataset) {
    if (!dataset) {
        return;
    }
    
    // Free name
    if (dataset->name) {
        free((void*)dataset->name);
    }
    
    // Free data if owned
    if (dataset->owns_data && dataset->data) {
        free(dataset->data);
    }
    
    // Free iterator state
    if (dataset->iter_state) {
        DatasetIterState* iter_state = (DatasetIterState*)dataset->iter_state;
        
        if (iter_state->indices) {
            free(iter_state->indices);
        }
        
        pthread_mutex_destroy(&iter_state->mutex);
        free(iter_state);
    }
    
    // Clear the struct
    memset(dataset, 0, sizeof(Dataset));
}

// Fisher-Yates shuffle
static void shuffle_indices(size_t* indices, size_t size, size_t seed) {
    srand(seed);
    
    for (size_t i = size - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        // Swap indices[i] and indices[j]
        size_t temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

// Reset dataset iterator
int dataset_reset(Dataset* dataset, bool shuffle) {
    if (!dataset || !dataset->iter_state) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    DatasetIterState* iter_state = (DatasetIterState*)dataset->iter_state;
    
    pthread_mutex_lock(&iter_state->mutex);
    
    // Reset current index
    iter_state->current_index = 0;
    
    // Shuffle indices if requested
    if (shuffle) {
        shuffle_indices(iter_state->indices, dataset->size, dataset->iter_config.seed);
    } else {
        // Reset to sequential order
        for (size_t i = 0; i < dataset->size; i++) {
            iter_state->indices[i] = i;
        }
    }
    
    pthread_mutex_unlock(&iter_state->mutex);
    
    return ONEBIT_SUCCESS;
}

// NOTE: The actual dataset_next_batch, dataset_load, dataset_split, and
// tokenization functions would be more complex and depend on the specific
// data formats and model requirements. This is a simplified placeholder implementation.

// Get next batch from dataset
int dataset_next_batch(Dataset* dataset, DataBatch* batch) {
    if (!dataset || !batch || !dataset->iter_state) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    DatasetIterState* iter_state = (DatasetIterState*)dataset->iter_state;
    
    pthread_mutex_lock(&iter_state->mutex);
    
    // Check if we've reached the end
    if (iter_state->current_index >= dataset->size) {
        pthread_mutex_unlock(&iter_state->mutex);
        return ONEBIT_ERROR_END_OF_DATA;
    }
    
    // Calculate batch size
    size_t remaining = dataset->size - iter_state->current_index;
    size_t requested_batch_size = dataset->iter_config.batch_size;
    size_t actual_batch_size = (remaining < requested_batch_size) ? remaining : requested_batch_size;
    
    // Check if we should drop the last incomplete batch
    if (dataset->iter_config.drop_last && actual_batch_size < requested_batch_size) {
        pthread_mutex_unlock(&iter_state->mutex);
        return ONEBIT_ERROR_END_OF_DATA;
    }
    
    // Set batch size in output
    batch->batch_size = actual_batch_size;
    
    // TODO: Load actual data into batch
    // This would depend on the specific data format and implementation.
    // For now, this is a placeholder that just advances the iterator.
    
    // Move to next batch
    iter_state->current_index += actual_batch_size;
    
    pthread_mutex_unlock(&iter_state->mutex);
    
    return ONEBIT_SUCCESS;
}

// Load a dataset from a file or directory
int dataset_load(const char* path, const DataIterConfig* config, Dataset** dataset) {
    if (!path || !config || !dataset) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // TODO: Implement actual dataset loading from files
    // This would depend on the specific data format and implementation.
    
    // For now, return a placeholder error
    return ONEBIT_ERROR_NOT_IMPLEMENTED;
}

// Split a dataset into training and validation sets
int dataset_split(const Dataset* dataset, float train_ratio,
                 Dataset** train_dataset, Dataset** val_dataset, size_t seed) {
    if (!dataset || !train_dataset || !val_dataset || train_ratio <= 0.0f || train_ratio >= 1.0f) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // TODO: Implement dataset splitting
    // This would involve creating two new datasets and copying/linking data appropriately.
    
    // For now, return a placeholder error
    return ONEBIT_ERROR_NOT_IMPLEMENTED;
}

// Token sequence functions
int token_sequence_init(TokenSequence* sequence, size_t capacity) {
    if (!sequence || capacity == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Clear the sequence struct
    memset(sequence, 0, sizeof(TokenSequence));
    
    // Allocate tokens array
    sequence->tokens = (uint32_t*)malloc(capacity * sizeof(uint32_t));
    if (!sequence->tokens) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    sequence->capacity = capacity;
    sequence->length = 0;
    
    return ONEBIT_SUCCESS;
}

void token_sequence_cleanup(TokenSequence* sequence) {
    if (!sequence) {
        return;
    }
    
    if (sequence->tokens) {
        free(sequence->tokens);
    }
    
    memset(sequence, 0, sizeof(TokenSequence));
}

int token_sequence_add(TokenSequence* sequence, uint32_t token_id) {
    if (!sequence || !sequence->tokens) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Check if we need to resize
    if (sequence->length >= sequence->capacity) {
        size_t new_capacity = sequence->capacity * 2;
        uint32_t* new_tokens = (uint32_t*)realloc(sequence->tokens, 
                                                 new_capacity * sizeof(uint32_t));
        if (!new_tokens) {
            return ONEBIT_ERROR_MEMORY;
        }
        
        sequence->tokens = new_tokens;
        sequence->capacity = new_capacity;
    }
    
    // Add token
    sequence->tokens[sequence->length++] = token_id;
    
    return ONEBIT_SUCCESS;
}

// Tokenizer and vocabulary functions
// These are placeholders and would need actual implementations
// based on the specific tokenization approach used.

int tokenizer_init(Tokenizer* tokenizer, Vocabulary* vocab) {
    if (!tokenizer || !vocab) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    tokenizer->vocab = vocab;
    tokenizer->add_bos_token = false;
    tokenizer->add_eos_token = false;
    tokenizer->tokenizer_state = NULL;
    
    return ONEBIT_SUCCESS;
}

void tokenizer_cleanup(Tokenizer* tokenizer) {
    if (!tokenizer) {
        return;
    }
    
    // Just clear the struct, don't free vocab as it's managed separately
    tokenizer->vocab = NULL;
    
    // Free tokenizer state if it exists
    if (tokenizer->tokenizer_state) {
        free(tokenizer->tokenizer_state);
        tokenizer->tokenizer_state = NULL;
    }
}

int vocab_load(const char* path, Vocabulary** vocab) {
    if (!path || !vocab) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // TODO: Implement loading vocabulary from file
    
    return ONEBIT_ERROR_NOT_IMPLEMENTED;
}

int vocab_create(const char** tokens, size_t size, Vocabulary** vocab) {
    if (!tokens || size == 0 || !vocab) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Allocate vocabulary struct
    *vocab = (Vocabulary*)malloc(sizeof(Vocabulary));
    if (!*vocab) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Clear the struct
    memset(*vocab, 0, sizeof(Vocabulary));
    
    // Allocate tokens array
    (*vocab)->tokens = (const char**)malloc(size * sizeof(char*));
    if (!(*vocab)->tokens) {
        free(*vocab);
        *vocab = NULL;
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Copy token pointers
    for (size_t i = 0; i < size; i++) {
        (*vocab)->tokens[i] = strdup(tokens[i]);
        if (!(*vocab)->tokens[i]) {
            // Free previously allocated tokens
            for (size_t j = 0; j < i; j++) {
                free((void*)(*vocab)->tokens[j]);
            }
            free((void*)(*vocab)->tokens);
            free(*vocab);
            *vocab = NULL;
            return ONEBIT_ERROR_MEMORY;
        }
    }
    
    (*vocab)->size = size;
    
    // TODO: Build encoding trie or lookup for efficient tokenization
    
    return ONEBIT_SUCCESS;
}

void vocab_cleanup(Vocabulary* vocab) {
    if (!vocab) {
        return;
    }
    
    // Free tokens
    if (vocab->tokens) {
        for (size_t i = 0; i < vocab->size; i++) {
            if (vocab->tokens[i]) {
                free((void*)vocab->tokens[i]);
            }
        }
        free((void*)vocab->tokens);
    }
    
    // Free special tokens
    if (vocab->unk_token) free((void*)vocab->unk_token);
    if (vocab->pad_token) free((void*)vocab->pad_token);
    if (vocab->bos_token) free((void*)vocab->bos_token);
    if (vocab->eos_token) free((void*)vocab->eos_token);
    
    // Free encoding trie if it exists
    if (vocab->encoding_trie) {
        free(vocab->encoding_trie);
    }
    
    // Clear the struct
    memset(vocab, 0, sizeof(Vocabulary));
}

int tokenizer_encode(const Tokenizer* tokenizer, const char* text, TokenSequence* sequence) {
    if (!tokenizer || !tokenizer->vocab || !text || !sequence) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // TODO: Implement actual tokenization
    // This would depend on the specific tokenization approach.
    
    return ONEBIT_ERROR_NOT_IMPLEMENTED;
}

int tokenizer_decode(const Tokenizer* tokenizer, const TokenSequence* sequence, char** text) {
    if (!tokenizer || !tokenizer->vocab || !sequence || !text) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // TODO: Implement actual detokenization
    // This would depend on the specific tokenization approach.
    
    return ONEBIT_ERROR_NOT_IMPLEMENTED;
} 