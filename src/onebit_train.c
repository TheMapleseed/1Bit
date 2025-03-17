/**
 * @file onebit_train.c
 * @brief Implementation of high-performance training system
 */

#include "onebit/onebit_train.h"
#include "onebit/onebit_compute.h"
#include "onebit/onebit_memory.h"
#include "onebit/onebit_error.h"
#include "onebit/onebit_optimizer.h"
#include "onebit/onebit_metrics.h"
#include "onebit/onebit_concurrency.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

struct TrainingContext {
    TrainingConfig config;
    ComputeContext* compute;
    
    // Model state
    float* weights;
    float* gradients;
    float* adam_m;
    float* adam_v;
    
    // Training state
    size_t step_count;
    float current_loss;
    float current_accuracy;
    
    // Memory management
    size_t total_params;
    void* workspace;
};

// Additional internal structure for online learning
typedef struct {
    OnlineLearningConfig config;
    float* cached_inputs;
    float* cached_targets;
    int cache_count;
    int update_counter;
    float cumulative_loss;
    int total_updates;
    Mutex lock;  // Thread safety for online updates
    bool initialized;
} OnlineLearningState;

// Global state for online learning
static OnlineLearningState g_online_state = {0};

TrainingContext* train_init(const TrainingConfig* config) {
    TrainingContext* ctx = calloc(1, sizeof(TrainingContext));
    if (!ctx) return NULL;
    
    // Copy configuration
    memcpy(&ctx->config, config, sizeof(TrainingConfig));
    
    // Initialize compute context
    if (onebit_init() != 0) {
        free(ctx);
        return NULL;
    }
    
    // Allocate model state
    ctx->total_params = /* calculate total parameters */;
    size_t state_size = ctx->total_params * sizeof(float);
    
    ctx->weights = aligned_alloc(64, state_size);
    ctx->gradients = aligned_alloc(64, state_size);
    
    if (config->use_adam) {
        ctx->adam_m = aligned_alloc(64, state_size);
        ctx->adam_v = aligned_alloc(64, state_size);
        memset(ctx->adam_m, 0, state_size);
        memset(ctx->adam_v, 0, state_size);
    }
    
    // Initialize weights
    for (size_t i = 0; i < ctx->total_params; i++) {
        ctx->weights[i] = (float)rand() / RAND_MAX * 0.1f;
    }
    
    return ctx;
}

int train_step(TrainingContext* ctx, const float* input, const float* target) {
    // Forward pass
    Matrix input_mat = {
        .data = (void*)input,
        .rows = ctx->config.batch_size,
        .cols = /* input dimension */,
        .stride = /* input dimension */,
        .layout = MATRIX_LAYOUT_ROW_MAJOR
    };
    
    // Compute forward pass
    float loss = 0.0f;
    size_t correct = 0;
    
    // Backward pass
    memset(ctx->gradients, 0, ctx->total_params * sizeof(float));
    
    // Update metrics
    ctx->current_loss = loss;
    ctx->current_accuracy = (float)correct / ctx->config.batch_size;
    
    ctx->step_count++;
    return 0;
}

int train_update_params(TrainingContext* ctx) {
    float lr = ctx->config.learning_rate;
    
    // Learning rate warmup
    if (ctx->config.warmup_steps > 0 && 
        ctx->step_count < ctx->config.warmup_steps) {
        lr *= (float)ctx->step_count / ctx->config.warmup_steps;
    }
    
    if (ctx->config.use_adam) {
        // Adam update
        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float eps = 1e-8f;
        
        float beta1_t = powf(beta1, ctx->step_count);
        float beta2_t = powf(beta2, ctx->step_count);
        
        for (size_t i = 0; i < ctx->total_params; i++) {
            ctx->adam_m[i] = beta1 * ctx->adam_m[i] + 
                            (1 - beta1) * ctx->gradients[i];
            ctx->adam_v[i] = beta2 * ctx->adam_v[i] + 
                            (1 - beta2) * ctx->gradients[i] * ctx->gradients[i];
            
            float m_hat = ctx->adam_m[i] / (1 - beta1_t);
            float v_hat = ctx->adam_v[i] / (1 - beta2_t);
            
            ctx->weights[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }
    } else {
        // SGD update
        for (size_t i = 0; i < ctx->total_params; i++) {
            ctx->weights[i] -= lr * ctx->gradients[i];
        }
    }
    
    // Weight decay
    if (ctx->config.weight_decay > 0.0f) {
        for (size_t i = 0; i < ctx->total_params; i++) {
            ctx->weights[i] *= (1.0f - ctx->config.weight_decay);
        }
    }
    
    return 0;
}

void train_get_metrics(TrainingContext* ctx, float* loss, float* accuracy) {
    if (loss) *loss = ctx->current_loss;
    if (accuracy) *accuracy = ctx->current_accuracy;
}

void train_cleanup(TrainingContext* ctx) {
    if (!ctx) return;
    
    free(ctx->weights);
    free(ctx->gradients);
    free(ctx->adam_m);
    free(ctx->adam_v);
    free(ctx->workspace);
    
    onebit_cleanup();
    free(ctx);
}

// Initialize training
int onebit_train_init(OneBitContext* ctx, TrainConfig* config, TrainState** state) {
    if (!ctx || !config || !state) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    TrainState* train_state = (TrainState*)onebit_malloc(ctx, sizeof(TrainState));
    if (!train_state) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Clear the structure
    memset(train_state, 0, sizeof(TrainState));
    
    // Initialize optimizer
    // In a real implementation, this would create the appropriate optimizer
    // based on config parameters
    train_state->optimizer = optimizer_create(ctx, config->learning_rate, config->weight_decay);
    if (!train_state->optimizer) {
        onebit_free(ctx, train_state);
        return ONEBIT_ERROR_INITIALIZATION;
    }
    
    // Link to model
    train_state->model = ctx->model;
    if (!train_state->model) {
        optimizer_destroy(train_state->optimizer);
        onebit_free(ctx, train_state);
        return ONEBIT_ERROR_INVALID_MODEL;
    }
    
    // Initialize state variables
    train_state->global_step = 0;
    train_state->epoch = 0;
    train_state->best_loss = INFINITY;
    train_state->early_stopped = false;
    
    // Create checkpoint path if needed
    if (config->checkpoint_dir) {
        size_t path_len = strlen(config->checkpoint_dir) + 32;
        train_state->checkpoint_path = (char*)onebit_malloc(ctx, path_len);
        if (train_state->checkpoint_path) {
            snprintf(train_state->checkpoint_path, path_len, 
                    "%s/checkpoint", config->checkpoint_dir);
        }
    }
    
    *state = train_state;
    return ONEBIT_SUCCESS;
}

// Run full training loop
int onebit_train(OneBitContext* ctx, TrainState* state, Dataset* train_data, Dataset* val_data) {
    if (!ctx || !state || !train_data) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Set up data iterators
    state->train_data = dataset_iterator_create(train_data, ctx->config.batch_size);
    if (val_data) {
        state->valid_data = dataset_iterator_create(val_data, ctx->config.batch_size);
    }
    
    if (!state->train_data) {
        return ONEBIT_ERROR_DATASET;
    }
    
    // Run training loop
    for (state->epoch = 0; state->epoch < ctx->config.max_epochs; state->epoch++) {
        // Train one epoch
        int result = onebit_train_epoch(ctx, state);
        if (result != ONEBIT_SUCCESS) {
            return result;
        }
        
        // Validate if validation data is available
        if (state->valid_data) {
            float val_loss = onebit_validate(ctx, state);
            state->last_loss = val_loss;
            
            // Save checkpoint if best loss
            if (val_loss < state->best_loss) {
                state->best_loss = val_loss;
                if (state->checkpoint_path) {
                    onebit_save_checkpoint(ctx, state, state->checkpoint_path);
                }
            }
            
            // Early stopping logic would go here
        }
        
        // Custom checkpoint saving logic
        if (ctx->config.checkpoint_interval > 0 && 
            state->epoch % ctx->config.checkpoint_interval == 0 && 
            state->checkpoint_path) {
            
            char epoch_checkpoint[512];
            snprintf(epoch_checkpoint, sizeof(epoch_checkpoint), 
                    "%s_epoch%d", state->checkpoint_path, (int)state->epoch);
            onebit_save_checkpoint(ctx, state, epoch_checkpoint);
        }
        
        if (state->early_stopped) {
            break;
        }
    }
    
    return ONEBIT_SUCCESS;
}

// Train for a single epoch
int onebit_train_epoch(OneBitContext* ctx, TrainState* state) {
    if (!ctx || !state || !state->train_data) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Reset data iterator
    dataset_iterator_reset(state->train_data);
    
    float epoch_loss = 0.0f;
    int batch_count = 0;
    
    // Get batch size from the context
    size_t batch_size = ctx->config.batch_size;
    
    // Batch loop
    float* inputs = NULL;
    float* targets = NULL;
    
    while (dataset_iterator_next_batch(state->train_data, &inputs, &targets)) {
        // Zero gradients before forward pass
        model_zero_grad(state->model);
        
        // Forward pass
        float* outputs = onebit_malloc(ctx, 
                            batch_size * model_get_output_size(state->model) * sizeof(float));
        if (!outputs) {
            return ONEBIT_ERROR_MEMORY;
        }
        
        int result = model_forward(state->model, inputs, outputs, batch_size, 1);
        if (result != ONEBIT_SUCCESS) {
            onebit_free(ctx, outputs);
            return result;
        }
        
        // Compute loss and gradients
        float loss = 0.0f;
        result = compute_loss_and_gradients(ctx, outputs, targets, batch_size, 
                                         model_get_output_size(state->model), &loss);
        
        onebit_free(ctx, outputs);
        
        if (result != ONEBIT_SUCCESS) {
            return result;
        }
        
        // Backward pass
        result = model_backward(state->model, NULL); // Loss module computes initial gradients
        if (result != ONEBIT_SUCCESS) {
            return result;
        }
        
        // Optimizer step
        result = optimizer_step(state->optimizer, state->model);
        if (result != ONEBIT_SUCCESS) {
            return result;
        }
        
        // Update state
        state->global_step++;
        epoch_loss += loss;
        batch_count++;
        
        // Logging
        if (ctx->config.print_frequency > 0 && 
            state->global_step % ctx->config.print_frequency == 0) {
            // Log progress here
        }
    }
    
    if (batch_count > 0) {
        epoch_loss /= batch_count;
    }
    
    return ONEBIT_SUCCESS;
}

// Run validation
float onebit_validate(OneBitContext* ctx, TrainState* state) {
    if (!ctx || !state || !state->valid_data) {
        return INFINITY;
    }
    
    // Reset data iterator
    dataset_iterator_reset(state->valid_data);
    
    float total_loss = 0.0f;
    int batch_count = 0;
    
    // Get batch size
    size_t batch_size = ctx->config.batch_size;
    
    // Batch loop
    float* inputs = NULL;
    float* targets = NULL;
    
    while (dataset_iterator_next_batch(state->valid_data, &inputs, &targets)) {
        // Forward pass
        float* outputs = onebit_malloc(ctx, 
                            batch_size * model_get_output_size(state->model) * sizeof(float));
        if (!outputs) {
            return INFINITY;
        }
        
        int result = model_forward(state->model, inputs, outputs, batch_size, 1);
        if (result != ONEBIT_SUCCESS) {
            onebit_free(ctx, outputs);
            return INFINITY;
        }
        
        // Compute loss
        float loss = 0.0f;
        result = compute_loss(ctx, outputs, targets, batch_size, 
                           model_get_output_size(state->model), &loss);
        
        onebit_free(ctx, outputs);
        
        if (result != ONEBIT_SUCCESS) {
            return INFINITY;
        }
        
        total_loss += loss;
        batch_count++;
    }
    
    if (batch_count > 0) {
        total_loss /= batch_count;
    }
    
    return total_loss;
}

// Save checkpoint
int onebit_save_checkpoint(OneBitContext* ctx, TrainState* state, const char* path) {
    if (!ctx || !state || !path) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Save model parameters
    int result = model_save_weights(state->model, path);
    if (result != ONEBIT_SUCCESS) {
        return result;
    }
    
    // Save optimizer state if requested
    if (ctx->config.save_optimizer_state) {
        char optimizer_path[512];
        snprintf(optimizer_path, sizeof(optimizer_path), "%s.opt", path);
        result = optimizer_save_state(state->optimizer, optimizer_path);
        if (result != ONEBIT_SUCCESS) {
            return result;
        }
    }
    
    // Save training metadata
    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s.meta", path);
    
    FILE* meta_file = fopen(meta_path, "wb");
    if (!meta_file) {
        return ONEBIT_ERROR_FILE_OPEN;
    }
    
    // Write metadata
    fprintf(meta_file, "epoch=%zu\n", state->epoch);
    fprintf(meta_file, "global_step=%zu\n", state->global_step);
    fprintf(meta_file, "best_loss=%f\n", state->best_loss);
    fprintf(meta_file, "last_loss=%f\n", state->last_loss);
    
    fclose(meta_file);
    return ONEBIT_SUCCESS;
}

// Load checkpoint
int onebit_load_checkpoint(OneBitContext* ctx, TrainState* state, const char* path) {
    if (!ctx || !state || !path) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Load model parameters
    int result = model_load_weights(state->model, path);
    if (result != ONEBIT_SUCCESS) {
        return result;
    }
    
    // Load optimizer state if exists
    char optimizer_path[512];
    snprintf(optimizer_path, sizeof(optimizer_path), "%s.opt", path);
    
    FILE* opt_file = fopen(optimizer_path, "rb");
    if (opt_file) {
        fclose(opt_file);
        result = optimizer_load_state(state->optimizer, optimizer_path);
        if (result != ONEBIT_SUCCESS) {
            return result;
        }
    }
    
    // Load training metadata
    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s.meta", path);
    
    FILE* meta_file = fopen(meta_path, "rb");
    if (meta_file) {
        // Read metadata
        char line[256];
        while (fgets(line, sizeof(line), meta_file)) {
            if (strncmp(line, "epoch=", 6) == 0) {
                state->epoch = strtoull(line + 6, NULL, 10);
            } else if (strncmp(line, "global_step=", 12) == 0) {
                state->global_step = strtoull(line + 12, NULL, 10);
            } else if (strncmp(line, "best_loss=", 10) == 0) {
                state->best_loss = strtof(line + 10, NULL);
            } else if (strncmp(line, "last_loss=", 10) == 0) {
                state->last_loss = strtof(line + 10, NULL);
            }
        }
        
        fclose(meta_file);
    }
    
    return ONEBIT_SUCCESS;
}

// Clean up training state
void onebit_train_cleanup(TrainState* state) {
    if (!state) return;
    
    // Clean up optimizer
    if (state->optimizer) {
        optimizer_destroy(state->optimizer);
    }
    
    // Clean up scheduler if exists
    if (state->scheduler) {
        // scheduler_destroy(state->scheduler);
    }
    
    // Clean up data iterators
    if (state->train_data) {
        dataset_iterator_destroy(state->train_data);
    }
    
    if (state->valid_data) {
        dataset_iterator_destroy(state->valid_data);
    }
    
    // Free checkpoint path
    if (state->checkpoint_path) {
        free(state->checkpoint_path);
    }
    
    // Free the state itself
    free(state);
}

// Initialize and enable online learning
int onebit_enable_online_learning(OneBitContext* ctx, OnlineLearningConfig* config) {
    if (!ctx || !config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Initialize state if not already done
    if (!g_online_state.initialized) {
        memset(&g_online_state, 0, sizeof(OnlineLearningState));
        mutex_init(&g_online_state.lock);
        g_online_state.initialized = true;
    }
    
    mutex_lock(&g_online_state.lock);
    
    // Copy configuration
    memcpy(&g_online_state.config, config, sizeof(OnlineLearningConfig));
    
    // Reset counters
    g_online_state.update_counter = 0;
    g_online_state.total_updates = 0;
    g_online_state.cumulative_loss = 0.0f;
    g_online_state.cache_count = 0;
    
    // Allocate cache if needed
    if (config->cache_examples && config->cache_size > 0) {
        size_t input_size = model_get_input_size(ctx->model);
        size_t output_size = model_get_output_size(ctx->model);
        
        // Free existing cache if it exists
        if (g_online_state.cached_inputs) {
            onebit_free(ctx, g_online_state.cached_inputs);
        }
        
        if (g_online_state.cached_targets) {
            onebit_free(ctx, g_online_state.cached_targets);
        }
        
        // Allocate new cache
        g_online_state.cached_inputs = (float*)onebit_malloc(ctx, 
                                     config->cache_size * input_size * sizeof(float));
        g_online_state.cached_targets = (float*)onebit_malloc(ctx, 
                                      config->cache_size * output_size * sizeof(float));
        
        if (!g_online_state.cached_inputs || !g_online_state.cached_targets) {
            // Clean up on failure
            if (g_online_state.cached_inputs) {
                onebit_free(ctx, g_online_state.cached_inputs);
                g_online_state.cached_inputs = NULL;
            }
            
            if (g_online_state.cached_targets) {
                onebit_free(ctx, g_online_state.cached_targets);
                g_online_state.cached_targets = NULL;
            }
            
            mutex_unlock(&g_online_state.lock);
            return ONEBIT_ERROR_MEMORY;
        }
    }
    
    mutex_unlock(&g_online_state.lock);
    return ONEBIT_SUCCESS;
}

// Update model from feedback during or after inference
int onebit_online_update(OneBitContext* ctx, const float* input, const float* target, float* loss) {
    if (!ctx || !input || !target) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    // Check if online learning is enabled
    if (!g_online_state.initialized || !g_online_state.config.enabled) {
        return ONEBIT_ERROR_NOT_INITIALIZED;
    }
    
    mutex_lock(&g_online_state.lock);
    
    // Update counter
    g_online_state.update_counter++;
    
    // Add to cache if enabled
    if (g_online_state.config.cache_examples && g_online_state.cached_inputs && g_online_state.cached_targets) {
        size_t input_size = model_get_input_size(ctx->model);
        size_t output_size = model_get_output_size(ctx->model);
        
        if (g_online_state.cache_count < g_online_state.config.cache_size) {
            // Add to cache
            float* cache_input = g_online_state.cached_inputs + g_online_state.cache_count * input_size;
            float* cache_target = g_online_state.cached_targets + g_online_state.cache_count * output_size;
            
            memcpy(cache_input, input, input_size * sizeof(float));
            memcpy(cache_target, target, output_size * sizeof(float));
            
            g_online_state.cache_count++;
        } else {
            // Cache is full, replace random item
            int idx = rand() % g_online_state.config.cache_size;
            
            float* cache_input = g_online_state.cached_inputs + idx * input_size;
            float* cache_target = g_online_state.cached_targets + idx * output_size;
            
            memcpy(cache_input, input, input_size * sizeof(float));
            memcpy(cache_target, target, output_size * sizeof(float));
        }
    }
    
    // Decide whether to update now based on frequency
    bool should_update = g_online_state.config.update_frequency <= 1 || 
                        (g_online_state.update_counter % g_online_state.config.update_frequency == 0);
    
    if (should_update) {
        float current_loss = 0.0f;
        int result = ONEBIT_SUCCESS;
        
        if (g_online_state.config.cache_examples && g_online_state.cache_count > 1) {
            // Batch update from cache
            result = perform_batch_update(ctx, g_online_state.cached_inputs, 
                                       g_online_state.cached_targets, 
                                       g_online_state.cache_count, &current_loss);
        } else {
            // Single example update
            result = perform_single_update(ctx, input, target, &current_loss);
        }
        
        if (result == ONEBIT_SUCCESS) {
            g_online_state.cumulative_loss += current_loss;
            g_online_state.total_updates++;
            
            if (loss) {
                *loss = current_loss;
            }
        } else {
            mutex_unlock(&g_online_state.lock);
            return result;
        }
    }
    
    mutex_unlock(&g_online_state.lock);
    return ONEBIT_SUCCESS;
}

// Reset online learning state (e.g., cached examples)
void onebit_reset_online_learning(OneBitContext* ctx) {
    if (!ctx || !g_online_state.initialized) {
        return;
    }
    
    mutex_lock(&g_online_state.lock);
    
    g_online_state.cache_count = 0;
    g_online_state.update_counter = 0;
    g_online_state.cumulative_loss = 0.0f;
    g_online_state.total_updates = 0;
    
    mutex_unlock(&g_online_state.lock);
}

// Get online learning statistics
int onebit_get_online_learning_stats(OneBitContext* ctx, float* loss, float* learning_rate, int* updates) {
    if (!ctx || !g_online_state.initialized) {
        return ONEBIT_ERROR_NOT_INITIALIZED;
    }
    
    mutex_lock(&g_online_state.lock);
    
    if (loss) {
        if (g_online_state.total_updates > 0) {
            *loss = g_online_state.cumulative_loss / g_online_state.total_updates;
        } else {
            *loss = 0.0f;
        }
    }
    
    if (learning_rate) {
        *learning_rate = g_online_state.config.learning_rate;
    }
    
    if (updates) {
        *updates = g_online_state.total_updates;
    }
    
    mutex_unlock(&g_online_state.lock);
    return ONEBIT_SUCCESS;
}

// Internal helper: Perform update on a single example
static int perform_single_update(OneBitContext* ctx, const float* input, const float* target, float* loss) {
    // Zero gradients
    model_zero_grad(ctx->model);
    
    // Forward pass
    float* output = onebit_malloc(ctx, model_get_output_size(ctx->model) * sizeof(float));
    if (!output) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    int result = model_forward(ctx->model, input, output, 1, 1);
    if (result != ONEBIT_SUCCESS) {
        onebit_free(ctx, output);
        return result;
    }
    
    // Compute loss and gradients
    result = compute_loss_and_gradients(ctx, output, target, 1, 
                                     model_get_output_size(ctx->model), loss);
    
    onebit_free(ctx, output);
    
    if (result != ONEBIT_SUCCESS) {
        return result;
    }
    
    // Backward pass
    result = model_backward(ctx->model, NULL);
    if (result != ONEBIT_SUCCESS) {
        return result;
    }
    
    // Scale learning rate for online updates
    float original_lr = optimizer_get_learning_rate(ctx->optimizer);
    optimizer_set_learning_rate(ctx->optimizer, g_online_state.config.learning_rate);
    
    // Update model parameters with small step
    result = optimizer_step(ctx->optimizer, ctx->model);
    
    // Restore original learning rate
    optimizer_set_learning_rate(ctx->optimizer, original_lr);
    
    return result;
}

// Internal helper: Perform batch update from cache
static int perform_batch_update(OneBitContext* ctx, const float* inputs, 
                              const float* targets, int batch_size, float* loss) {
    // Zero gradients
    model_zero_grad(ctx->model);
    
    // Process each example
    float total_loss = 0.0f;
    
    size_t input_size = model_get_input_size(ctx->model);
    size_t output_size = model_get_output_size(ctx->model);
    
    for (int i = 0; i < batch_size; i++) {
        const float* current_input = inputs + i * input_size;
        const float* current_target = targets + i * output_size;
        
        // Forward pass
        float* output = onebit_malloc(ctx, output_size * sizeof(float));
        if (!output) {
            return ONEBIT_ERROR_MEMORY;
        }
        
        int result = model_forward(ctx->model, current_input, output, 1, 1);
        if (result != ONEBIT_SUCCESS) {
            onebit_free(ctx, output);
            return result;
        }
        
        // Compute loss and add to gradients (accumulate without zeroing)
        float example_loss = 0.0f;
        result = compute_loss_and_gradients(ctx, output, current_target, 1, 
                                         output_size, &example_loss);
        
        onebit_free(ctx, output);
        
        if (result != ONEBIT_SUCCESS) {
            return result;
        }
        
        total_loss += example_loss;
    }
    
    // Average loss
    if (batch_size > 0) {
        total_loss /= batch_size;
    }
    
    if (loss) {
        *loss = total_loss;
    }
    
    // Backward pass (already accumulated gradients)
    int result = model_backward(ctx->model, NULL);
    if (result != ONEBIT_SUCCESS) {
        return result;
    }
    
    // Scale learning rate for online updates
    float original_lr = optimizer_get_learning_rate(ctx->optimizer);
    optimizer_set_learning_rate(ctx->optimizer, g_online_state.config.learning_rate);
    
    // Update model parameters with small step
    result = optimizer_step(ctx->optimizer, ctx->model);
    
    // Restore original learning rate
    optimizer_set_learning_rate(ctx->optimizer, original_lr);
    
    return result;
} 