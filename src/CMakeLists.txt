cmake_minimum_required(VERSION 3.10)

# Core source files
set(ONEBIT_CORE_SOURCES
    onebit.c
    onebit_model.c
    onebit_tokenizer.c
    onebit_memory.c
    onebit_error.c
    onebit_utils.c
    onebit_transformer.c
    onebit_kv_cache.c
)

# CPU optimization sources
set(ONEBIT_CPU_SOURCES
    onebit_compute.c
    onebit_compute_cpu.c
    onebit_simd_avx512.c
    onebit_simd_avx2.c
    onebit_simd_sse4.c
)

# CUDA sources (conditionally included)
set(ONEBIT_CUDA_SOURCES "")
if(ONEBIT_USE_CUDA)
    set(ONEBIT_CUDA_SOURCES
        onebit_compute_cuda.cu
        onebit_cuda_kernels.cu
    )
    
    # Enable CUDA language
    enable_language(CUDA)
    
    # Set CUDA specific flags
    set_source_files_properties(${ONEBIT_CUDA_SOURCES} PROPERTIES
        CUDA_SOURCE_PROPERTY_FORMAT OBJ
        CUDA_ARCHITECTURES "70;75;80;86"
    )
endif()

# ROCm sources (conditionally included)
set(ONEBIT_ROCM_SOURCES "")
if(ONEBIT_USE_ROCM)
    set(ONEBIT_ROCM_SOURCES
        onebit_compute_rocm.cpp
        onebit_rocm_kernels.cpp
    )
endif()

# Metal sources (conditionally included for Apple platforms)
set(ONEBIT_METAL_SOURCES "")
if(APPLE AND ONEBIT_USE_METAL)
    set(ONEBIT_METAL_SOURCES
        onebit_compute_metal.m
        onebit_metal_kernels.m
    )
endif()

# Create the main OneBit library
add_library(onebit_compute SHARED
    ${ONEBIT_CORE_SOURCES}
    ${ONEBIT_CPU_SOURCES}
    ${ONEBIT_CUDA_SOURCES}
    ${ONEBIT_ROCM_SOURCES}
    ${ONEBIT_METAL_SOURCES}
)

# Set include directories
target_include_directories(onebit_compute
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link with required libraries
target_link_libraries(onebit_compute PRIVATE Threads::Threads)

# CUDA-specific dependencies
if(ONEBIT_USE_CUDA)
    find_package(CUDA REQUIRED)
    target_include_directories(onebit_compute PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(onebit_compute PRIVATE 
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES}
    )
endif()

# ROCm-specific dependencies
if(ONEBIT_USE_ROCM)
    target_link_libraries(onebit_compute PRIVATE hip::host hip::device)
endif()

# Metal-specific dependencies (for Apple platforms)
if(APPLE AND ONEBIT_USE_METAL)
    target_link_libraries(onebit_compute PRIVATE
        ${METAL_LIBRARY}
        ${FOUNDATION_LIBRARY}
    )
endif()

# Set properties for library
set_target_properties(onebit_compute PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

# Create alias for use from other directories
add_library(onebit::compute ALIAS onebit_compute)

# Install the target
install(TARGETS onebit_compute
    EXPORT OneBitTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Export targets
install(EXPORT OneBitTargets
    FILE OneBitTargets.cmake
    NAMESPACE onebit::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/onebit
)

# If examples are enabled, add that directory
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# If tests are enabled, add that directory
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
