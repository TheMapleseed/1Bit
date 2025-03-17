#include "onebit/onebit_optimize.h"
#include "onebit/onebit_error.h"
#include <math.h>
#include <string.h>

// Internal optimizer state
typedef struct {
    float* m;  // First moment
    float* v;  // Second moment
    size_t size;
} AdamState;

int optimizer_init(OptimizerContext* ctx, OptimizerType type,
                  const OptimizerConfig* config) {
    if (!ctx || !config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->type = type;
    memcpy(&ctx->config, config, sizeof(OptimizerConfig));
    ctx->step = 0;
    
    switch (type) {
        case OPTIMIZER_SGD:
            // No additional state needed
            ctx->state = NULL;
            break;
            
        case OPTIMIZER_ADAM: {
            AdamState* state = malloc(sizeof(AdamState));
            if (!state) {
                return ONEBIT_ERROR_MEMORY;
            }
            
            state->size = config->param_size;
            state->m = calloc(config->param_size, sizeof(float));
            state->v = calloc(config->param_size, sizeof(float));
            
            if (!state->m || !state->v) {
                free(state->m);
                free(state->v);
                free(state);
                return ONEBIT_ERROR_MEMORY;
            }
            
            ctx->state = state;
            break;
        }
        
        default:
            return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    return ONEBIT_SUCCESS;
}

void optimizer_cleanup(OptimizerContext* ctx) {
    if (!ctx) return;
    
    if (ctx->type == OPTIMIZER_ADAM) {
        AdamState* state = (AdamState*)ctx->state;
        free(state->m);
        free(state->v);
        free(state);
    }
}

int optimizer_step(OptimizerContext* ctx, float* params,
                  const float* grads) {
    if (!ctx || !params || !grads) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->step++;
    
    switch (ctx->type) {
        case OPTIMIZER_SGD: {
            float lr = ctx->config.learning_rate;
            
            #pragma omp parallel for
            for (size_t i = 0; i < ctx->config.param_size; i++) {
                params[i] -= lr * grads[i];
            }
            break;
        }
        
        case OPTIMIZER_ADAM: {
            AdamState* state = (AdamState*)ctx->state;
            float lr = ctx->config.learning_rate;
            float beta1 = ctx->config.beta1;
            float beta2 = ctx->config.beta2;
            float epsilon = ctx->config.epsilon;
            
            // Bias correction terms
            float bc1 = 1.0f - powf(beta1, ctx->step);
            float bc2 = 1.0f - powf(beta2, ctx->step);
            float lr_t = lr * sqrtf(bc2) / bc1;
            
            #pragma omp parallel for
            for (size_t i = 0; i < state->size; i++) {
                // Update bi
            }
            break;
        }
        
        default:
            return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    return ONEBIT_SUCCESS;
} 