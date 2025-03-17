cmake_minimum_required(VERSION 3.14)

# Examples for OneBit library
add_executable(onebit_inference inference.c)
target_link_libraries(onebit_inference PRIVATE onebit_compute)

add_executable(onebit_train train.c)
target_link_libraries(onebit_train PRIVATE onebit_compute)

add_executable(onebit_quantize quantize.c)
target_link_libraries(onebit_quantize PRIVATE onebit_compute)

# Install examples
install(TARGETS 
    onebit_inference
    onebit_train
    onebit_quantize
    RUNTIME DESTINATION bin/examples
) 