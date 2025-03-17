#include "onebit/onebit_metrics.h"
#include "onebit/onebit_error.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float compute_accuracy(const float* predictions,
                            const float* targets,
                            size_t batch_size,
                            float threshold) {
    size_t correct = 0;
    for (size_t i = 0; i < batch_size; i++) {
        bool pred = predictions[i] >= threshold;
        bool target = targets[i] >= threshold;
        if (pred == target) correct++;
    }
    return (float)correct / batch_size;
}

static void update_confusion_matrix(MetricContext* ctx,
                                  const float* predictions,
                                  const float* targets,
                                  size_t batch_size) {
    const size_t num_classes = ctx->config.num_classes;
    
    for (size_t i = 0; i < batch_size; i++) {
        size_t pred_class = 0;
        size_t true_class = 0;
        
        // Find predicted and true classes
        float max_pred = predictions[i * num_classes];
        float max_target = targets[i * num_classes];
        
        for (size_t j = 1; j < num_classes; j++) {
            if (predictions[i * num_classes + j] > max_pred) {
                max_pred = predictions[i * num_classes + j];
                pred_class = j;
            }
            if (targets[i * num_classes + j] > max_target) {
                max_target = targets[i * num_classes + j];
                true_class = j;
            }
        }
        
        // Update confusion matrix
        ctx->confusion_matrix[true_class * num_classes + pred_class]++;
    }
}

static float compute_precision(const size_t* confusion_matrix,
                             size_t num_classes,
                             size_t class_idx) {
    size_t true_positives = confusion_matrix[class_idx * num_classes + class_idx];
    size_t predicted_positives = 0;
    
    for (size_t i = 0; i < num_classes; i++) {
        predicted_positives += confusion_matrix[i * num_classes + class_idx];
    }
    
    return predicted_positives > 0 ? 
           (float)true_positives / predicted_positives : 0.0f;
}

static float compute_recall(const size_t* confusion_matrix,
                          size_t num_classes,
                          size_t class_idx) {
    size_t true_positives = confusion_matrix[class_idx * num_classes + class_idx];
    size_t actual_positives = 0;
    
    for (size_t i = 0; i < num_classes; i++) {
        actual_positives += confusion_matrix[class_idx * num_classes + i];
    }
    
    return actual_positives > 0 ? 
           (float)true_positives / actual_positives : 0.0f;
}

static float compute_f1(float precision, float recall) {
    return (precision + recall > 0.0f) ? 
           2.0f * precision * recall / (precision + recall) : 0.0f;
}

int metric_init(MetricContext* ctx, const MetricConfig* config) {
    if (!ctx || !config) return ONEBIT_ERROR_INVALID_PARAM;
    
    memset(ctx, 0, sizeof(MetricContext));
    memcpy(&ctx->config, config, sizeof(MetricConfig));
    
    if (config->num_classes > 0) {
        size_t matrix_size = config->num_classes * config->num_classes;
        ctx->confusion_matrix = calloc(matrix_size, sizeof(size_t));
        ctx->class_metrics = calloc(config->num_classes, sizeof(float));
        
        if (!ctx->confusion_matrix || !ctx->class_metrics) {
            metric_cleanup(ctx);
            return ONEBIT_ERROR_MEMORY;
        }
    }
    
    ctx->initialized = true;
    return ONEBIT_SUCCESS;
}

void metric_cleanup(MetricContext* ctx) {
    if (!ctx) return;
    
    free(ctx->confusion_matrix);
    free(ctx->class_metrics);
    memset(ctx, 0, sizeof(MetricContext));
}

int metric_update(MetricContext* ctx, const float* predictions,
                 const float* targets, size_t batch_size) {
    if (!ctx || !predictions || !targets || !ctx->initialized) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    switch (ctx->config.type) {
        case METRIC_ACCURACY:
            ctx->running_sum += compute_accuracy(predictions, targets,
                                              batch_size, ctx->config.threshold);
            break;
            
        case METRIC_PRECISION:
        case METRIC_RECALL:
        case METRIC_F1_SCORE:
            update_confusion_matrix(ctx, predictions, targets, batch_size);
            break;
            
        case METRIC_MSE: {
            float sum = 0.0f;
            for (size_t i = 0; i < batch_size; i++) {
                float diff = predictions[i] - targets[i];
                sum += diff * diff;
            }
            ctx->running_sum += sum;
            break;
        }
        
        case METRIC_MAE: {
            float sum = 0.0f;
            for (size_t i = 0; i < batch_size; i++) {
                sum += fabsf(predictions[i] - targets[i]);
            }
            ctx->running_sum += sum;
            break;
        }
        
        // Additional metrics...
        
        default:
            return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    ctx->sample_count += batch_size;
    return ONEBIT_SUCCESS;
}

float metric_get_value(const MetricContext* ctx) {
    if (!ctx || !ctx->initialized || ctx->sample_count == 0) return 0.0f;
    
    switch (ctx->config.type) {
        case METRIC_ACCURACY:
            return ctx->running_sum / ctx->sample_count;
            
        case METRIC_PRECISION:
        case METRIC_RECALL:
        case METRIC_F1_SCORE: {
            float sum = 0.0f;
            for (size_t i = 0; i < ctx->config.num_classes; i++) {
                float precision = compute_precision(ctx->confusion_matrix,
                                                 ctx->config.num_classes, i);
                float recall = compute_recall(ctx->confusion_matrix,
                                           ctx->config.num_classes, i);
                
                switch (ctx->config.type) {
                    case METRIC_PRECISION:
                        sum += precision;
                        break;
                    case METRIC_RECALL:
                        sum += recall;
                        break;
                    case METRIC_F1_SCORE:
                        sum += compute_f1(precision, recall);
                        break;
                    default:
                        break;
                }
            }
            return sum / ctx->config.num_classes;
        }
        
        case METRIC_MSE:
        case METRIC_MAE:
            return ctx->running_sum / ctx->sample_count;
            
        default:
            return 0.0f;
    }
}

// ... (remaining implementation of other functions) 