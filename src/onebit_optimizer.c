#include "onebit/onebit_optimizer.h"
#include "onebit/onebit_error.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void apply_sgd(OptimizerContext* ctx, float* params, const float* gradients) {
    const float lr = ctx->config.learning_rate;
    
    for (size_t i = 0; i < ctx->state.param_size; i++) {
        params[i] -= lr * gradients[i];
    }
}

static void apply_momentum(OptimizerContext* ctx, float* params, const float* gradients) {
    const float lr = ctx->config.learning_rate;
    const float momentum = ctx->config.momentum;
    const bool nesterov = ctx->config.nesterov;
    float* momentum_buffer = ctx->state.momentum_buffer;
    
    for (size_t i = 0; i < ctx->state.param_size; i++) {
        momentum_buffer[i] = momentum * momentum_buffer[i] + gradients[i];
        
        if (nesterov) {
            params[i] -= lr * (gradients[i] + momentum * momentum_buffer[i]);
        } else {
            params[i] -= lr * momentum_buffer[i];
        }
    }
}

static void apply_rmsprop(OptimizerContext* ctx, float* params, const float* gradients) {
    const float lr = ctx->config.learning_rate;
    const float beta2 = ctx->config.beta2;
    const float epsilon = ctx->config.epsilon;
    float* square_avg = ctx->state.square_avg_buffer;
    
    for (size_t i = 0; i < ctx->state.param_size; i++) {
        square_avg[i] = beta2 * square_avg[i] + (1 - beta2) * gradients[i] * gradients[i];
        params[i] -= lr * gradients[i] / (sqrtf(square_avg[i]) + epsilon);
    }
}

static void apply_adam(OptimizerContext* ctx, float* params, const float* gradients) {
    const float lr = ctx->config.learning_rate;
    const float beta1 = ctx->config.beta1;
    const float beta2 = ctx->config.beta2;
    const float epsilon = ctx->config.epsilon;
    float* momentum_avg = ctx->state.momentum_avg_buffer;
    float* square_avg = ctx->state.square_avg_buffer;
    
    ctx->state.iteration++;
    
    // Compute bias correction terms
    float bias_correction1 = 1.0f - powf(beta1, ctx->state.iteration);
    float bias_correction2 = 1.0f - powf(beta2, ctx->state.iteration);
    
    for (size_t i = 0; i < ctx->state.param_size; i++) {
        // Update momentum and RMSprop moving averages
        momentum_avg[i] = beta1 * momentum_avg[i] + (1 - beta1) * gradients[i];
        square_avg[i] = beta2 * square_avg[i] + (1 - beta2) * gradients[i] * gradients[i];
        
        // Compute bias-corrected estimates
        float momentum_corrected = momentum_avg[i] / bias_correction1;
        float square_corrected = square_avg[i] / bias_correction2;
        
        // Update parameters
        params[i] -= lr * momentum_corrected / (sqrtf(square_corrected) + epsilon);
    }
}

static void apply_adamw(OptimizerContext* ctx, float* params, const float* gradients) {
    const float lr = ctx->config.learning_rate;
    const float beta1 = ctx->config.beta1;
    const float beta2 = ctx->config.beta2;
    const float epsilon = ctx->config.epsilon;
    const float weight_decay = ctx->config.weight_decay;
    float* momentum_avg = ctx->state.momentum_avg_buffer;
    float* square_avg = ctx->state.square_avg_buffer;
    
    ctx->state.iteration++;
    
    // Compute bias correction terms
    float bias_correction1 = 1.0f - powf(beta1, ctx->state.iteration);
    float bias_correction2 = 1.0f - powf(beta2, ctx->state.iteration);
    
    for (size_t i = 0; i < ctx->state.param_size; i++) {
        // Update momentum and RMSprop moving averages
        momentum_avg[i] = beta1 * momentum_avg[i] + (1 - beta1) * gradients[i];
        square_avg[i] = beta2 * square_avg[i] + (1 - beta2) * gradients[i] * gradients[i];
        
        // Compute bias-corrected estimates
        float momentum_corrected = momentum_avg[i] / bias_correction1;
        float square_corrected = square_avg[i] / bias_correction2;
        
        // Update parameters
        params[i] -= lr * momentum_corrected / (sqrtf(square_corrected) + epsilon);
    }
} 