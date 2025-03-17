#include "onebit/onebit_hardware.h"
#include "onebit/onebit_error.h"
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/sysinfo.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

HardwareType detect_hardware(void) {
    // Check for CUDA
    #ifdef ONEBIT_USE_CUDA
    int cuda_devices = 0;
    if (cudaGetDeviceCount(&cuda_devices) == cudaSuccess && cuda_devices > 0) {
        return HARDWARE_CUDA;
    }
    #endif
    
    // Check for Metal
    #ifdef __APPLE__
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device) {
        return HARDWARE_METAL;
    }
    #endif
    
    return HARDWARE_CPU;
}

int get_hardware_info(HardwareInfo* info) {
    if (!info) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    memset(info, 0, sizeof(HardwareInfo));
    info->type = detect_hardware();
    
    switch (info->type) {
        case HARDWARE_CPU:
            #ifdef _WIN32
            SYSTEM_INFO sysInfo;
            GetSystemInfo(&sysInfo);
            info->compute_units = sysInfo.dwNumberOfProcessors;
            #else
            info->compute_units = sysconf(_SC_NPROCESSORS_ONLN);
            #endif
            
            #ifdef __APPLE__
            size_t len = sizeof(info->device_name);
            sysctlbyname("machdep.cpu.brand_string", info->device_name, &len, NULL, 0);
            #else
            strncpy(info->device_name, "Generic CPU", sizeof(info->device_name) - 1);
            #endif
            break;
            
        case HARDWARE_CUDA:
            #ifdef ONEBIT_USE_CUDA
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            info->compute_units = prop.multiProcessorCount;
            info->memory_size = prop.totalGlobalMem;
            info->compute_capability = prop.major * 10 + prop.minor;
            strncpy(info->device_name, prop.name, sizeof(info->device_name) - 1);
            info->supports_fp16 = (info->compute_capability >= 60);
            info->supports_int8 = (info->compute_capability >= 61);
            #endif
            break;
            
        case HARDWARE_METAL:
            #ifdef __APPLE__
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device) {
                info->compute_units = 1;
                strncpy(info->device_name, "Apple M1", sizeof(info->device_name) - 1);
            }
            #endif
            break;
    }
    
    return ONEBIT_SUCCESS;
} 