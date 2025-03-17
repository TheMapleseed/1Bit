#include "onebit/onebit_random.h"
#include "onebit/onebit_error.h"
#include <string.h>
#include <time.h>

// PCG random number generator state
typedef struct {
    uint64_t state;
    uint64_t inc;
} PCGState;

int random_init(RandomContext* ctx, const RandomConfig* config) {
    if (!ctx) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    PCGState* state = malloc(sizeof(PCGState));
    if (!state) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Initialize PCG state
    uint64_t seed = config ? config->seed : time(NULL);
    state->state = seed + 1442695040888963407ULL;
    state->inc = (seed << 1) | 1;
    
    ctx->state = state;
    
    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        free(state);
        return ONEBIT_ERROR_THREAD;
    }
    
    return ONEBIT_SUCCESS;
}

void random_cleanup(RandomContext* ctx) {
    if (!ctx) return;
    
    free(ctx->state);
    pthread_mutex_destroy(&ctx->mutex);
}

static uint32_t pcg32(PCGState* state) {
    uint64_t oldstate = state->state;
    state->state = oldstate * 6364136223846793005ULL + state->inc;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

uint32_t random_uint32(RandomContext* ctx) {
    if (!ctx) return 0;
    
    pthread_mutex_lock(&ctx->mutex);
    uint32_t result = pcg32((PCGState*)ctx->state);
    pthread_mutex_unlock(&ctx->mutex);
    
    return result;
}

uint64_t random_uint64(RandomContext* ctx) {
    if (!ctx) return 0;
    
    pthread_mutex_lock(&ctx->mutex);
    uint64_t result = ((uint64_t)pcg32((PCGState*)ctx->state) << 32) |
                      pcg32((PCGState*)ctx->state);
    pthread_mutex_unlock(&ctx->mutex);
    
    return result;
}

float random_float(RandomContext* ctx) {
    if (!ctx) return 0.0f;
    
    pthread_mutex_lock(&ctx->mutex);
    uint32_t bits = pcg32((PCGState*)ctx->state);
    pthread_mutex_unlock(&ctx->mutex);
    
    // Convert to float in [0, 1)
    return (bits >> 8) * 0x1.0p-24f;
}

double random_double(RandomContext* ctx) {
    if (!ctx) return 0.0;
    
    pthread_mutex_lock(&ctx->mutex);
    uint64_t bits = ((uint64_t)pcg32((PCGState*)ctx->state) << 32) |
                    pcg32((PCGState*)ctx->state);
    pthread_mutex_unlock(&ctx->mutex);
    
    // Convert to double in [0, 1)
    return (bits >> 11) * 0x1.0p-53;
}

void random_bytes(RandomContext* ctx, void* buffer, size_t size) {
    if (!ctx || !buffer) return;
    
    uint8_t* bytes = (uint8_t*)buffer;
    size_t remaining = size;
    
    pthread_mutex_lock(&ctx->mutex);
    
    while (remaining > 0) {
        uint32_t value = pcg32((PCGState*)ctx->state);
        size_t chunk = remaining < 4 ? remaining : 4;
        memcpy(bytes, &value, chunk);
        bytes += chunk;
        remaining -= chunk;
    }
    
    pthread_mutex_unlock(&ctx->mutex);
}

int32_t random_int32_range(RandomContext* ctx, int32_t min,
                          int32_t max) {
    if (!ctx || min > max) return 0;
    
    uint32_t range = (uint32_t)(max - min + 1);
    uint32_t limit = -range % range;
    uint32_t value;
    
    pthread_mutex_lock(&ctx->mutex);
    
    do {
        value = pcg32((PCGState*)ctx->state);
    } while (value < limit);
    
    pthread_mutex_unlock(&ctx->mutex);
    
    return min + (value % range);
}

float random_float_range(RandomContext* ctx, float min,
                        float max) {
    if (!ctx) return 0.0f;
    return 
} 