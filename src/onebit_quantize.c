#include "onebit/onebit_quantize.h"
#include "onebit/onebit_error.h"
#include <math.h>
#include <string.h>

// Internal quantization helpers
static float compute_scale(const float* data, size_t size) {
    float max_abs = 0.0f;
    
    for (size_t i = 0; i < size; i++) {
        float abs_val = fabsf(data[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }
    
    return max_abs / 127.0f;  // For int8 quantization
}

int quantize_tensor(const float* input, void* output,
                   size_t size, QuantizeType type,
                   QuantizationParams* params) {
    if (!input || !output || !params) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    switch (type) {
        case QUANTIZE_INT8: {
            int8_t* out = (int8_t*)output;
            params->scale = compute_scale(input, size);
            float inv_scale = 1.0f / params->scale;
            
            for (size_t i = 0; i < size; i++) {
                float scaled = input[i] * inv_scale;
                out[i] = (int8_t)roundf(fmaxf(-127.0f,
                                             fminf(127.0f, scaled)));
            }
            break;
        }
        
        case QUANTIZE_INT4: {
            uint8_t* out = (uint8_t*)output;
            params->scale = compute_scale(input, size);
            float inv_scale = 1.0f / params->scale;
            
            for (size_t i = 0; i < size; i += 2) {
                float scaled1 = input[i] * inv_scale;
                float scaled2 = (i + 1 < size) ? input[i + 1] * inv_scale : 0.0f;
                
                int4_t val1 = (int4_t)roundf(fmaxf(-7.0f,
                                                  fminf(7.0f, scaled1)));
                int4_t val2 = (int4_t)roundf(fmaxf(-7.0f,
                                                  fminf(7.0f, scaled2)));
                
                out[i] = (val1 << 4) | val2;
            }
            break;
        }
        
        default:
            return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    return ONEBIT_SUCCESS;
} 