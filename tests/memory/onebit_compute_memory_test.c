#include "../test_utils.h"
#include "onebit_compute.h"
#include <stdlib.h>
#include <string.h>

// Memory tracking
typedef struct {
    void* ptr;
    size_t size;
    const char* file;
    int line;
} MemoryAllocation;

#define MAX_ALLOCATIONS 1000
static MemoryAllocation g_allocations[MAX_ALLOCATIONS];
static size_t g_num_allocations = 0;
static size_t g_total_allocated = 0;

// Memory tracking functions
void* tracked_malloc(size_t size, const char* file, int line) {
    void* ptr = malloc(size);
    if (ptr && g_num_allocations < MAX_ALLOCATIONS) {
        g_allocations[g_num_allocations++] = (MemoryAllocation){
            .ptr = ptr,
            .size = size,
            .file = file,
            .line = line
        };
        g_total_allocated += size;
    }
    return ptr;
}

void tracked_free(void* ptr) {
    if (!ptr) return;
    
    for (size_t i = 0; i < g_num_allocations; i++) {
        if (g_allocations[i].ptr == ptr) {
            g_total_allocated -= g_allocations[i].size;
            memmove(&g_allocations[i], &g_allocations[i + 1],
                   (g_num_allocations - i - 1) * sizeof(MemoryAllocation));
            g_num_allocations--;
            break;
        }
    }
    free(ptr);
}

// Memory leak test cases
void test_compute_context_memory(TestContext* test) {
    g_num_allocations = 0;
    g_total_allocated = 0;
    
    // Test context creation/destruction
    ComputeContext* ctx;
    compute_init(&ctx, COMPUTE_DEVICE_CPU);
    
    size_t after_init = g_total_allocated;
    assert_test(test, "Context Allocation",
                after_init > 0 && g_num_allocations > 0);
    
    compute_cleanup(ctx);
    assert_test(test, "Context Cleanup",
                g_total_allocated == 0 && g_num_allocations == 0);
}

void test_kernel_memory_leaks(TestContext* test) {
    g_num_allocations = 0;
} 