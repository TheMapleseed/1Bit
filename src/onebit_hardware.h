#ifndef ONEBIT_HARDWARE_H
#define ONEBIT_HARDWARE_H

#include <stdbool.h>

typedef enum {
    HARDWARE_CPU = 0,
    HARDWARE_CUDA = 1,
    HARDWARE_METAL = 2,
    HARDWARE_UNKNOWN = -1
} HardwareType;

typedef struct {
    HardwareType type;
    char device_name[256];
    size_t compute_units;
    size_t memory_size;
    bool supports_fp16;
    bool supports_int8;
    int compute_capability;  // For CUDA
    char* vendor;
    char* architecture;
} HardwareInfo;

// Function declarations
HardwareType detect_hardware(void);
int get_hardware_info(HardwareInfo* info);
bool is_hardware_supported(HardwareType type);
const char* hardware_type_string(HardwareType type);

#endif 