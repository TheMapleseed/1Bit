#include "onebit/onebit_base64.h"
#include "onebit/onebit_error.h"
#include <string.h>

static const char base64_chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static const unsigned char base64_decode_table[256] = {
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 62, 64, 64, 64, 63,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 64, 64, 64, 64, 64,
    64,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 64, 64, 64, 64, 64,
    64, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64
};

size_t base64_encode_size(size_t input_size) {
    return ((input_size + 2) / 3) * 4;
}

size_t base64_decode_size(size_t input_size) {
    return ((input_size + 3) / 4) * 3;
}

int base64_encode(const void* input, size_t input_size,
                 char* output, size_t output_size) {
    if (!input || !output || input_size == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    size_t required_size = base64_encode_size(input_size);
    if (output_size < required_size + 1) {
        return ONEBIT_ERROR_BUFFER_OVERFLOW;
    }
    
    const unsigned char* data = (const unsigned char*)input;
    char* out = output;
    size_t i = 0;
    
    // Encode each 3-byte chunk
    for (; i + 2 < input_size; i += 3) {
        *out++ = base64_chars[(data[i] >> 2) & 0x3F];
        *out++ = base64_chars[((data[i] & 0x3) << 4) |
                             ((data[i + 1] >> 4) & 0xF)];
        *out++ = base64_chars[((data[i + 1] & 0xF) << 2) |
                             ((data[i + 2] >> 6) & 0x3)];
        *out++ = base64_chars[data[i + 2] & 0x3F];
    }
    
    // Handle remaining bytes
    if (i < input_size) {
        *out++ = base64_chars[(data[i] >> 2) & 0x3F];
        if (i + 1 < input_size) {
            *out++ = base64_chars[((data[i] & 0x3) << 4) |
                                 ((data[i + 1] >> 4) & 0xF)];
            *out++ = base64_chars[(data[i + 1] & 0xF) << 2];
            *out++ = '=';
        } else {
            *out++ = base64_chars[(data[i] & 0x3) << 4];
            *out++ = '=';
            *out++ = '=';
        }
    }
    
    *out = '\0';
    return ONEBIT_SUCCESS;
}

int base64_decode(const char* input, size_t input_size,
                 void* output, size_t output_size,
                 size_t* decoded_size) {
    if (!input || !output || !decoded_size || input_size == 0 ||
        (input_size % 4) != 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    size_t required_size = base64_decode_size(input_size);
    if (output_size < required_size) {
        return ONEBIT_ERROR_BUFFER_OVERFLOW;
    }
    
    unsigned char* out = (unsigned char*)output;
    size_t out_pos = 0;
    unsigned char block[4];
    
    for (size_t i = 0; i < input_size; i += 4) {
        // Get values for this block
        for (size_t j = 0; j < 4; j++) {
            unsigned char val = base64_decode_table[(unsigned char)input[i + j]];
            if (val == 64) {
                if (j < 2 || (j == 2 && input[i + 3] != '=')) {
                    return ONEBIT_ERROR_INVALID_FORMAT;
                }
                block[j] = 0;
            } else {
                block[j] = val;
            }
        }
        
        // Decode block
        out[out_pos++] = (block[0] << 2) | (block[1] >> 4);
        if (input[i + 2] != '=') {
            out[out_pos++] = (block[1] << 4) | (block[2] >> 2);
            if (input[i + 3] != '=') {
                out[out_pos++] = (block[2] << 6) | block[3];
            }
        }
    }
    
    *decoded_size = out_pos;
    return ONEBIT_SUCCESS;
}

bool base64_is_valid(const char* input, size_t input_size) {
    if (!input || input_size == 0 || (input_size % 4) != 0) {
        return false;
    }
    
    size_t padding = 0;
    for (size_t i = 0; i < input_size; i++) {
        if (input[i] == '=') {
            padding++;
            if (padding > 2 || i < input_size - 2) {
                return false;
            }
        } else if (padding > 0) {
            return false;
        } else if (base64_decode_table[(unsigned char)input[i]] == 64) {
            return false;
        }
    }
    
    return true;
} 