#include "onebit/onebit_compute.h"
#include "onebit/onebit_device_manager.h"
#include <string.h>
#include <stdlib.h>

// Device capabilities structure
typedef struct {
    ComputeDeviceType type;
    size_t memory_size;
    int compute_units;
    char name[256];
    float performance_score;
    bool is_available;
} DeviceInfo;

// Global device manager state
typedef struct {
    DeviceInfo* devices;
    size_t num_devices;
    ComputeDeviceType active_device;
    char last_error[256];
} DeviceManager;

static DeviceManager g_device_manager = {0};

// Device detection functions
static bool detect_cuda_devices(DeviceInfo* devices, size_t* count) {
    #ifdef HAVE_CUDA
    int cuda_count = 0;
    cudaError_t err = cudaGetDeviceCount(&cuda_count);
    
    if (err == cudaSuccess && cuda_count > 0) {
        for (int i = 0; i < cuda_count && *count < MAX_DEVICES; i++) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                devices[*count].type = COMPUTE_DEVICE_CUDA;
                devices[*count].memory_size = prop.totalGlobalMem;
                devices[*count].compute_units = prop.multiProcessorCount;
                strncpy(devices[*count].name, prop.name, sizeof(devices[*count].name) - 1);
                devices[*count].is_available = true;
                
                // Calculate performance score based on compute capabilities
                devices[*count].performance_score = 
                    prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor * 
                    (prop.clockRate / 1000.0f);
                
                (*count)++;
            }
        }
        return true;
    }
    #endif
    return false;
}

static bool detect_metal_devices(DeviceInfo* devices, size_t* count) {
    #ifdef __APPLE__
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device) {
        devices[*count].type = COMPUTE_DEVICE_METAL;
        devices[*count].memory_size = [device recommendedMaxWorkingSetSize];
        devices[*count].compute_units = device.maxThreadsPerThreadgroup;
        strncpy(devices[*count].name, 
                [[device name] UTF8String], 
                sizeof(devices[*count].name) - 1);
        devices[*count].is_available = true;
        
        // Calculate performance score
        devices[*count].performance_score = 
            device.maxThreadsPerThreadgroup * 
            (float)[device maxTransferRate] / 1000.0f;
        
        (*count)++;
        return true;
    }
    #endif
    return false;
}

static void detect_cpu_device(DeviceInfo* devices, size_t* count) {
    devices[*count].type = COMPUTE_DEVICE_CPU;
    devices[*count].compute_units = sysconf(_SC_NPROCESSORS_ONLN);
    devices[*count].is_available = true;
    
    #ifdef _SC_PHYS_PAGES
    devices[*count].memory_size = 
        (size_t)sysconf(_SC_PHYS_PAGES) * 
        (size_t)sysconf(_SC_PAGE_SIZE);
    #else
    devices[*count].memory_size = 8ULL * 1024 * 1024 * 1024; // 8GB default
    #endif
    
    strcpy(devices[*count].name, "CPU");
    
    // Calculate CPU performance score
    devices[*count].performance_score = 
        devices[*count].compute_units * 
        get_cpu_base_clock_speed();
    
    (*count)++;
}

// Initialize device manager
int device_manager_init(void) {
    // Check if already initialized
    if (g_device_manager.devices) {
        ONEBIT_SET_ERROR(ONEBIT_ERROR_ALREADY_INITIALIZED);
        return ONEBIT_ERROR_ALREADY_INITIALIZED;
    }
    
    // Allocate device array
    g_device_manager.devices = calloc(MAX_DEVICES, sizeof(DeviceInfo));
    if (!g_device_manager.devices) {
        ONEBIT_SET_ERROR(ONEBIT_ERROR_MEMORY);
        return ONEBIT_ERROR_MEMORY;
    }
    
    size_t count = 0;
    bool any_device_found = false;
    
    // Try CUDA initialization
    if (detect_cuda_devices(g_device_manager.devices, &count)) {
        any_device_found = true;
    } else {
        // Log CUDA initialization failure if CUDA is available
        #ifdef HAVE_CUDA
        ONEBIT_SET_ERROR_MSG(ONEBIT_ERROR_CUDA_INIT, 
                            cudaGetErrorString(cudaGetLastError()));
        #endif
    }
    
    // Try Metal initialization
    if (!any_device_found && detect_metal_devices(g_device_manager.devices, &count)) {
        any_device_found = true;
    } else {
        // Log Metal initialization failure if Metal is available
        #ifdef __APPLE__
        ONEBIT_SET_ERROR(ONEBIT_ERROR_METAL_INIT);
        #endif
    }
    
    // Fallback to CPU
    if (!any_device_found) {
        detect_cpu_device(g_device_manager.devices, &count);
        if (count == 0) {
            ONEBIT_SET_ERROR(ONEBIT_ERROR_DEVICE_NOT_FOUND);
            free(g_device_manager.devices);
            g_device_manager.devices = NULL;
            return ONEBIT_ERROR_DEVICE_NOT_FOUND;
        }
    }
    
    g_device_manager.num_devices = count;
    
    // Select best device
    float best_score = 0;
    bool device_selected = false;
    
    for (size_t i = 0; i < count; i++) {
        if (g_device_manager.devices[i].is_available && 
            g_device_manager.devices[i].performance_score > best_score) {
            best_score = g_device_manager.devices[i].performance_score;
            g_device_manager.active_device = g_device_manager.devices[i].type;
            device_selected = true;
        }
    }
    
    if (!device_selected) {
        ONEBIT_SET_ERROR(ONEBIT_ERROR_INITIALIZATION);
        free(g_device_manager.devices);
        g_device_manager.devices = NULL;
        return ONEBIT_ERROR_INITIALIZATION;
    }
    
    onebit_clear_error();
    return ONEBIT_SUCCESS;
}

// Get active device
ComputeDeviceType device_manager_get_active(void) {
    return g_device_manager.active_device;
}

// Set active device
int device_manager_set_active(ComputeDeviceType type) {
    for (size_t i = 0; i < g_device_manager.num_devices; i++) {
        if (g_device_manager.devices[i].type == type && 
            g_device_manager.devices[i].is_available) {
            g_device_manager.active_device = type;
            return ONEBIT_SUCCESS;
        }
    }
    return ONEBIT_ERROR_INVALID_DEVICE;
}

// Get device info
const DeviceInfo* device_manager_get_info(ComputeDeviceType type) {
    for (size_t i = 0; i < g_device_manager.num_devices; i++) {
        if (g_device_manager.devices[i].type == type) {
            return &g_device_manager.devices[i];
        }
    }
    return NULL;
}

// Cleanup
void device_manager_cleanup(void) {
    free(g_device_manager.devices);
    memset(&g_device_manager, 0, sizeof(DeviceManager));
} 
} 