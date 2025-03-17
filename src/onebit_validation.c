#include "onebit/onebit_validation.h"
#include "onebit/onebit_error.h"
#include <math.h>
#include <string.h>

int validation_init(ValidationContext* ctx, const ValidationConfig* config) {
    if (!ctx || !config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->metrics = malloc(config->max_metrics * sizeof(ValidationMetric));
    if (!ctx->metrics) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    ctx->num_metrics = 0;
    ctx->max_metrics = config->max_metrics;
    ctx->best_metric = 0.0f;
    ctx->patience = config->patience;
    ctx->patience_counter = 0;
    ctx->mode = config->mode;
    
    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        free(ctx->metrics);
        return ONEBIT_ERROR_THREAD;
    }
    
    return ONEBIT_SUCCESS;
}

void validation_cleanup(ValidationContext* ctx) {
    if (!ctx) return;
    
    pthread_mutex_lock(&ctx->mutex);
    
    for (size_t i = 0; i < ctx->num_metrics; i++) {
        free(ctx->metrics[i].name);
    }
    
    free(ctx->metrics);
    
    pthread_mutex_unlock(&ctx->mutex);
    pthread_mutex_destroy(&ctx->mutex);
}

int validation_add_metric(ValidationContext* ctx, const char* name,
                         ValidationMetricType type) {
    if (!ctx || !name) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&ctx->mutex);
    
    if (ctx->num_metrics >= ctx->max_metrics) {
        pthread_mutex_unlock(&ctx->mutex);
        return ONEBIT_ERROR_OVERFLOW;
    }
    
    // Check if metric already exists
    for (size_t i = 0; i < ctx->num_metrics; i++) {
        if (strcmp(ctx->metrics[i].name, name) == 0) {
            pthread_mutex_unlock(&ctx->mutex);
            return ONEBIT_ERROR_INVALID_PARAM;
        }
    }
    
    ValidationMetric* metric = &ctx->metrics[ctx->num_metrics];
    
    metric->name = strdup(name);
    if (!metric->name) {
        pthread_mutex_unlock(&ctx->mutex);
        return ONEBIT_ERROR_MEMORY;
    }
    
    metric->type = type;
    metric->value = 0.0f;
    metric->best_value = ctx->best_metric;
    
    pthread_mutex_unlock(&ctx->mutex);
    return ONEBIT_SUCCESS;
} 