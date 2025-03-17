cmake_minimum_required(VERSION 3.14)
project(onebit VERSION 1.0.0 LANGUAGES C CXX)

# Options
option(ONEBIT_USE_CUDA "Enable CUDA support" OFF)
option(ONEBIT_USE_ROCM "Enable ROCm support" OFF)
option(ONEBIT_USE_METAL "Enable Metal support for Apple platforms" OFF)
option(ONEBIT_ENABLE_AVX512 "Enable AVX-512 optimizations" ON)
option(ONEBIT_ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(ONEBIT_ENABLE_SSE4 "Enable SSE4 optimizations" ON)
option(ONEBIT_BUILD_TESTS "Build test programs" ON)
option(ONEBIT_BUILD_EXAMPLES "Build example programs" ON)

# Check hardware capabilities
include(CheckCXXCompilerFlag)
if(ONEBIT_ENABLE_AVX512)
    check_cxx_compiler_flag(-mavx512f AVX512_SUPPORTED)
    if(AVX512_SUPPORTED)
        add_compile_options(-mavx512f)
        add_definitions(-DONEBIT_ENABLE_AVX512)
    else()
        message(STATUS "AVX-512 not supported by compiler, disabling")
        set(ONEBIT_ENABLE_AVX512 OFF)
    endif()
endif()

if(ONEBIT_ENABLE_AVX2)
    check_cxx_compiler_flag(-mavx2 AVX2_SUPPORTED)
    if(AVX2_SUPPORTED)
        add_compile_options(-mavx2)
        add_definitions(-DONEBIT_ENABLE_AVX2)
    else()
        message(STATUS "AVX2 not supported by compiler, disabling")
        set(ONEBIT_ENABLE_AVX2 OFF)
    endif()
endif()

if(ONEBIT_ENABLE_SSE4)
    check_cxx_compiler_flag(-msse4.1 SSE4_SUPPORTED)
    if(SSE4_SUPPORTED)
        add_compile_options(-msse4.1)
        add_definitions(-DONEBIT_ENABLE_SSE4)
    else()
        message(STATUS "SSE4 not supported by compiler, disabling")
        set(ONEBIT_ENABLE_SSE4 OFF)
    endif()
endif()

# Set up threading
find_package(Threads REQUIRED)

# CUDA configuration
if(ONEBIT_USE_CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DONEBIT_USE_CUDA)
    
    set(CUDA_ARCH_FLAGS "-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS}")
    
    # Set C++11 standard for CUDA
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# ROCm configuration
if(ONEBIT_USE_ROCM)
    find_package(hip REQUIRED)
    add_definitions(-DONEBIT_USE_ROCM)
endif()

# Metal configuration (for Apple platforms)
if(APPLE AND ONEBIT_USE_METAL)
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    add_definitions(-DONEBIT_USE_METAL)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Create include directory and subdirectories
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/onebit)

# Add subdirectories
add_subdirectory(src)

if(ONEBIT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ONEBIT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Install configuration
include(GNUInstallDirs)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ 
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Generate and install config files
include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/OneBitConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/OneBitConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/onebit
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/OneBitConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/OneBitConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/OneBitConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/onebit
) 