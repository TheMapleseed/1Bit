#ifndef ONEBIT_COMPUTE_H
#define ONEBIT_COMPUTE_H

#include "onebit_memory.h"
#include <stdint.h>

// Compute capabilities flags
#define COMPUTE_CAP_FP16     (1 << 0)
#define COMPUTE_CAP_INT8     (1 << 1)
#define COMPUTE_CAP_TENSOR   (1 << 2)
#define COMPUTE_CAP_UNIFIED  (1 << 3)

// Operation types
typedef enum {
    OP_MATMUL,
    OP_LAYERNORM,
    OP_SOFTMAX,
    OP_ATTENTION,
    OP_GELU,
    OP_DROPOUT
} OpType;

// Compute descriptor
typedef struct {
    OpType op;
    void* inputs[4];
    void* outputs[4];
    int num_inputs;
    int num_outputs;
    uint32_t flags;
    void* workspace;
    size_t workspace_size;
} ComputeDesc;

// Compute context
typedef struct {
    void* device_ctx;
    uint32_t capabilities;
    int device_id;
    void* stream;
    void* workspace;
    size_t workspace_size;
} ComputeContext;

// Function declarations
int compute_init(ComputeContext* ctx, int device_id);
void compute_cleanup(ComputeContext* ctx);

// Operation execution
int compute_execute(ComputeContext* ctx, ComputeDesc* desc);
int compute_sync(ComputeContext* ctx);

// Workspace management
size_t compute_workspace_size(ComputeContext* ctx, ComputeDesc* desc);
void* compute_workspace_alloc(ComputeContext* ctx, size_t size);
void compute_workspace_free(ComputeContext* ctx, void* workspace);

// Performance monitoring
typedef struct {
    uint64_t op_count;         // Number of operations executed
    double compute_time;       // Total compute time
    double memory_transfer;    // Memory transfer time
    double utilization;        // Device utilization
    size_t memory_used;       // Current memory usage
    int active_streams;       // Number of active streams
} ComputeStats;

int compute_get_stats(ComputeContext* ctx, ComputeStats* stats); 