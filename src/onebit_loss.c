#include "onebit/onebit_loss.h"
#include "onebit/onebit_error.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float compute_mse(const float* predictions, const float* targets,
                        float* gradients, size_t batch_size, bool compute_grad) {
    float loss = 0.0f;
    
    for (size_t i = 0; i < batch_size; i++) {
        float diff = predictions[i] - targets[i];
        loss += diff * diff;
        
        if (compute_grad && gradients) {
            gradients[i] = 2.0f * diff;
        }
    }
    
    return loss / (float)batch_size;
}

static float compute_mae(const float* predictions, const float* targets,
                        float* gradients, size_t batch_size, bool compute_grad) {
    float loss = 0.0f;
    
    for (size_t i = 0; i < batch_size; i++) {
        float diff = predictions[i] - targets[i];
        loss += fabsf(diff);
        
        if (compute_grad && gradients) {
            gradients[i] = diff > 0.0f ? 1.0f : -1.0f;
        }
    }
    
    return loss / (float)batch_size;
}

static float compute_huber(LossContext* ctx, const float* predictions,
                          const float* targets, float* gradients,
                          size_t batch_size, bool compute_grad) {
    float loss = 0.0f;
    float delta = ctx->config.delta;
    
    for (size_t i = 0; i < batch_size; i++) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        
        if (abs_diff <= delta) {
            loss += 0.5f * diff * diff;
            if (compute_grad && gradients) {
                gradients[i] = diff;
            }
        } else {
            loss += delta * (abs_diff - 0.5f * delta);
            if (compute_grad && gradients) {
                gradients[i] = delta * (diff > 0.0f ? 1.0f : -1.0f);
            }
        }
    }
    
    return loss / (float)batch_size;
}

static float compute_cross_entropy(LossContext* ctx, const float* predictions,
                                 const float* targets, float* gradients,
                                 size_t batch_size, bool compute_grad) {
    float loss = 0.0f;
    float eps = ctx->config.eps;
    
    for (size_t i = 0; i < batch_size; i++) {
        float pred = fmaxf(fminf(predictions[i], 1.0f - eps), eps);
        loss -= targets[i] * logf(pred) + (1.0f - targets[i]) * logf(1.0f - pred);
        
        if (compute_grad && gradients) {
            gradients[i] = (pred - targets[i]) / (pred * (1.0f - pred));
        }
    }
    
    return loss / (float)batch_size;
} 