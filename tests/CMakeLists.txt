# Test executables
add_executable(test_onebit
    bench/onebit_compute_bench.c
    memory/onebit_compute_memory_gpu.c
    memory/onebit_compute_memory_stress.c
    arch/cpu_specific_test.c
    arch/cuda_specific_test.cu
)

if(APPLE)
    target_sources(test_onebit PRIVATE
        memory/onebit_compute_memory_metal.mm
        arch/metal_specific_test.mm
    )
endif()

target_link_libraries(test_onebit
    PRIVATE
        onebit_compute
        pthread
) 