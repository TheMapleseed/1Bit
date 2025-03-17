#include "bitnet.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

void test_cuda_initialization(void) {
    BitnetCUDAContext ctx;
    assert(init_cuda_context(&ctx, 0) == BITNET_SUCCESS);
    assert(ctx.cublas_handle != NULL);
    assert(ctx.stream != NULL);
    assert(ctx.workspace != NULL);
    assert(ctx.workspace_size > 0);
    
    destroy_cuda_context(&ctx);
    printf("CUDA initialization test passed\n");
}

void test_cuda_memory(void) {
    void* d_ptr = NULL;
    size_t test_size = 1024 * 1024;  // 1MB
    
    // Test allocation
    assert(cuda_malloc(&d_ptr, test_size) == BITNET_SUCCESS);
    assert(d_ptr != NULL);
    
    // Test host-to-device transfer
    float* h_data = malloc(test_size);
    for (size_t i = 0; i < test_size/sizeof(float); i++) {
        h_data[i] = (float)i;
    }
    
    assert(cuda_memcpy_to_device(d_ptr, h_data, test_size) == BITNET_SUCCESS);
    
    // Test device-to-host transfer
    float* h_result = malloc(test_size);
    assert(cuda_memcpy_to_host(h_result, d_ptr, test_size) == BITNET_SUCCESS);
    
    // Verify data
    for (size_t i = 0; i < test_size/sizeof(float); i++) {
        assert(fabs(h_data[i] - h_result[i]) < 1e-6);
    }
    
    cuda_free(d_ptr);
    free(h_data);
    free(h_result);
    printf("CUDA memory test passed\n");
}

void test_cuda_tensor_ops(void) {
    BitnetCUDAContext ctx;
    assert(init_cuda_context(&ctx, 0) == BITNET_SUCCESS);
    
    // Create test tensors
    uint32_t dims[] = {2, 3, 4, 1};
    CUDATensor A, B, C;
    
    assert(cuda_tensor_create(&A, dims, TENSOR_TYPE_FLOAT32, true) == BITNET_SUCCESS);
    assert(cuda_tensor_create(&B, dims, TENSOR_TYPE_FLOAT32, true) == BITNET_SUCCESS);
    assert(cuda_tensor_create(&C, dims, TENSOR_TYPE_FLOAT32, true) == BITNET_SUCCESS);
    
    // Initialize test data
    float* h_data = malloc(A.size);
    for (size_t i = 0; i < A.size/sizeof(float); i++) {
        h_data[i] = 1.0f;
    }
    
    assert(cuda_memcpy_to_device(A.data, h_data, A.size) == BITNET_SUCCESS);
    assert(cuda_memcpy_to_device(B.data, h_data, B.size) == BITNET_SUCCESS);
    
    // Test matrix multiplication
    assert(cuda_matrix_multiply(&ctx, &A, &B, &C, false, false) == BITNET_SUCCESS);
    
    // Verify result
    float* h_result = malloc(C.size);
    assert(cuda_memcpy_to_host(h_result, C.data, C.size) == BITNET_SUCCESS);
    
    // Check matrix multiplication result
    for (size_t i = 0; i < C.size/sizeof(float); i++) {
        assert(fabs(h_result[i] - 3.0f) < 1e-6);  // 3 = dot product of 1s
    }
    
    free(h_data);
    free(h_result);
    cuda_tensor_destroy(&A);
    cuda_tensor_destroy(&B);
    cuda_tensor_destroy(&C);
    destroy_cuda_context(&ctx);
    printf("CUDA tensor operations test passed\n");
}

void test_cuda_kernels(void) {
    BitnetCUDAContext ctx;
    assert(init_cuda_context(&ctx, 0) == BITNET_SUCCESS);
    
    // Test layer normalization
    uint32_t dims[] = {1, 768, 1, 1};  // typical hidden size
    CUDATensor input, weight, bias;
    
    assert(cuda_tensor_create(&input, dims, TENSOR_TYPE_FLOAT32, true) == BITNET_SUCCESS);
    assert(cuda_tensor_create(&weight, dims, TENSOR_TYPE_FLOAT32, true) == BITNET_SUCCESS);
    assert(cuda_tensor_create(&bias, dims, TENSOR_TYPE_FLOAT32, true) == BITNET_SUCCESS);
    
    // Initialize test data
    float* h_input = malloc(input.size);
    float* h_weight = malloc(weight.size);
    float* h_bias = malloc(bias.size);
    
    for (size_t i = 0; i < input.size/sizeof(float); i++) {
        h_input[i] = 1.0f;
        h_weight[i] = 1.0f;
        h_bias[i] = 0.0f;
    }
    
    assert(cuda_memcpy_to_device(input.data, h_input, input.size) == BITNET_SUCCESS);
    assert(cuda_memcpy_to_device(weight.data, h_weight, weight.size) == BITNET_SUCCESS);
    assert(cuda_memcpy_to_device(bias.data, h_bias, bias.size) == BITNET_SUCCESS);
    
    // Test layer normalization
    assert(cuda_layer_norm(&ctx, &input, &weight, &bias, 1e-5f) == BITNET_SUCCESS);
    
    // Verify result
    float* h_result = malloc(input.size);
    assert(cuda_memcpy_to_host(h_result, input.data, input.size) == BITNET_SUCCESS);
    
    // Check if normalized (mean ≈ 0, variance ≈ 1)
    float mean = 0.0f, var = 0.0f;
    size_t n = input.size/sizeof(float);
    
    for (size_t i = 0; i < n; i++) {
        mean += h_result[i];
    }
    mean /= n;
    
    for (size_t i = 0; i < n; i++) {
        var += (h_result[i] - mean) * (h_result[i] - mean);
    }
    var /= n;
    
    assert(fabs(mean) < 1e-6);
    assert(fabs(var - 1.0f) < 1e-6);
    
    free(h_input);
    free(h_weight);
    free(h_bias);
    free(h_result);
    cuda_tensor_destroy(&input);
    cuda_tensor_destroy(&weight);
    cuda_tensor_destroy(&bias);
    destroy_cuda_context(&ctx);
    printf("CUDA kernels test passed\n");
}

int main(void) {
    printf("Running CUDA tests...\n");
    
    test_cuda_initialization();
    test_cuda_memory();
    test_cuda_tensor_ops();
    test_cuda_kernels();
    
    printf("All CUDA tests passed!\n");
    return 0;
} 