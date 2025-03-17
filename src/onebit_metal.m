#include <metal_stdlib>
using namespace metal;

// Matrix multiplication kernel matching preset
kernel void matrix_multiply(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= N || gid.y >= M) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[gid.y * K + k] * B[k * N + gid.x];
    }
    C[gid.y * N + gid.x] = sum;
}

// Layer normalization kernel
kernel void layer_norm(
    device float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    constant int& size [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    // Compute mean
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += input[i];
    }
    float mean = sum / size;
    
    // Compute variance
    sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = input[i] - mean;
        sum += diff * diff;
    }
    float variance = sum / size;
    
    // Normalize
    input[gid] = gamma[gid] * (input[gid] - mean) / sqrt(variance + eps) + beta[gid];
}

// Attention computation kernel
kernel void attention_compute(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& num_heads [[buffer(5)]],
    constant int& seq_length [[buffer(6)]],
    constant int& head_dim [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int b = gid.x;
    const int h = gid.y;
    const int i = gid.z;
    
    if (b >= batch_size || h >= num_heads || i >= seq_length) return;
    
    // Compute attention scores
    float scores[1024];  // Assuming max seq length
    float sum = 0.0f;
    
    for (int j = 0; j < seq_length; ++j) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += query[b * num_heads * seq_length * head_dim + 
                         h * seq_length * head_dim +
                         i * head_dim + d] *
                    key[b * num_heads * seq_length * head_dim +
                        h * seq_length * head_dim +
                        j * head_dim + d];
        }
        score /= sqrt(float(head_dim));
        scores[j] = exp(score);
        sum += scores[j];
    }
    
    // Normalize and compute weighted sum
    for (int d = 0; d < head_dim; ++d) {
        float weighted_sum = 0.0f;
        for (int j = 0; j < seq_length; ++j) {
            weighted_sum += (scores[j] / sum) * 
                          value[b * num_heads * seq_length * head_dim +
                                h * seq_length * head_dim +
                                j * head_dim + d];
        }
        output[b * num_heads * seq_length * head_dim +
               h * seq_length * head_dim +
               i * head_dim + d] = weighted_sum;
    }
} 

// Add dynamic library export support
extern "C" {
    void* get_kernel_matrix_multiply() {
        return (__bridge void*)matrix_multiply;
    }
    
    // ... other kernel exports
} 