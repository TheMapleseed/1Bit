#include "onebit/onebit_io.h"
#include "onebit/onebit_error.h"
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif

int save_tensor(const char* filename, const QuantizedTensor* tensor) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        return ONEBIT_ERROR_IO;
    }
    
    // Write header
    OneBitFileHeader header;
    memcpy(header.magic, "ONEBIT\0\0", 8);
    header.version = 1;
    header.type = tensor->type;
    memcpy(header.dims, tensor->dims, sizeof(uint32_t) * 4);
    header.checksum = compute_checksum(tensor->data, tensor->size);
    
    if (write_header(fp, &header) != ONEBIT_SUCCESS) {
        fclose(fp);
        return ONEBIT_ERROR_IO;
    }
    
    // Write data
    if (fwrite(tensor->data, 1, tensor->size, fp) != tensor->size) {
        fclose(fp);
        return ONEBIT_ERROR_IO;
    }
    
    // Write scaling factors
    if (fwrite(tensor->scales, sizeof(float), tensor->num_scales, fp) != tensor->num_scales) {
        fclose(fp);
        return ONEBIT_ERROR_IO;
    }
    
    fclose(fp);
    return ONEBIT_SUCCESS;
}

int load_tensor(const char* filename, QuantizedTensor* tensor) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        return ONEBIT_ERROR_IO;
    }
    
    // Read and validate header
    OneBitFileHeader header;
    if (read_header(fp, &header) != ONEBIT_SUCCESS) {
        fclose(fp);
        return ONEBIT_ERROR_IO;
    }
    
    if (validate_header(&header) != ONEBIT_SUCCESS) {
        fclose(fp);
        return ONEBIT_ERROR_INVALID;
    }
    
    // Allocate memory
    size_t data_size = get_tensor_size(&header);
    tensor->data = malloc(data_size);
    if (!tensor->data) {
        fclose(fp);
        return ONEBIT_ERROR_MEMORY;
    }
    
    // Read data
    if (fread(tensor->data, 1, data_size, fp) != data_size) {
        fclose(fp);
        return ONEBIT_ERROR_IO;
    }
    
    // Read scaling factors
    if (fread(tensor->scales, sizeof(float), tensor->num_scales, fp) != tensor->num_scales) {
        fclose(fp);
        return ONEBIT_ERROR_IO;
    }
    
    fclose(fp);
    return ONEBIT_SUCCESS;
} 