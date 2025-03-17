#include "onebit/onebit_scheduler.h"
#include "onebit/onebit_error.h"
#include <math.h>
#include <string.h>

int scheduler_init(SchedulerContext* ctx, SchedulerType type,
                  const SchedulerConfig* config) {
    if (!ctx || !config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->type = type;
    memcpy(&ctx->config, config, sizeof(SchedulerConfig));
    ctx->step = 0;
    ctx->current_lr = config->initial_lr;
    
    return ONEBIT_SUCCESS;
}

float scheduler_step(SchedulerContext* ctx) {
    if (!ctx) {
        return 0.0f;
    }
    
    ctx->step++;
    
    switch (ctx->type) {
        case SCHEDULER_CONSTANT:
            // Learning rate remains constant
            break;
            
        case SCHEDULER_STEP: {
            // Decay learning rate by gamma every step_size steps
            int num_decays = ctx->step / ctx->config.step_size;
            ctx->current_lr = ctx->config.initial_lr *
                            powf(ctx->config.gamma, num_decays);
            break;
        }
        
        case SCHEDULER_EXPONENTIAL:
            // Exponential decay
            ctx->current_lr = ctx->config.initial_lr *
                            powf(ctx->config.gamma, ctx->step);
            break;
            
        case SCHEDULER_COSINE: {
            // Cosine annealing
            float progress = (float)ctx->step / ctx->config.max_steps;
            progress = fminf(1.0f, progress);
            
            ctx->current_lr = ctx->config.min_lr + 
                            0.5f * (ctx->config.initial_lr - ctx->config.min_lr) *
                            (1.0f + cosf(M_PI * progress));
            break;
        }
        
        case SCHEDULER_LINEAR: {
            // Linear decay
            float progress = (float)ctx->step / ctx->config.max_steps;
            progress = fminf(1.0f, progress);
            
            ctx->current_lr = ctx->config.initial_lr +
                            progress * (ctx->config.min_lr - ctx->config.initial_lr);
            break;
        }
        
        case SCHEDULER_WARMUP: {
            // Linear warmup followed by constant learning rate
            if (ctx->step < ctx->config.warmup_steps) {
                float progress = (float)ctx->step / ctx->config.warmup_steps;
                ctx->current_lr = ctx->config.initial_lr * progress;
            }
            break;
        }
    }
    
    return ctx->current_lr;
}

void scheduler_reset(SchedulerContext* ctx) {
    if (!ctx) return;
    
    ctx->step = 0;
    ctx->current_lr = ctx->config.initial_lr;
}

int scheduler_get_state(const SchedulerContext* ctx,
                       SchedulerState* state) {
    if (!ctx || !state) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    state->step = ctx->step;
    state->current_lr = ctx->current_lr;
    state->type = ctx->type;
    memcpy(&state->config, &ctx->config, sizeof(SchedulerConfig));
    
    return ONEBIT_SUCCESS;
}

int scheduler_set_state(SchedulerContext* ctx,
                       const SchedulerState* state) {
    if (!ctx || !state) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->step = state->step;
    ctx->current_lr = state->current_lr;
    ctx->type = state->type;
    memcpy(&ctx->config, &state->config, sizeof(SchedulerConfig));
    
    return ONEBIT_SUCCESS;
} 