cmake_minimum_required(VERSION 3.20)

project(distance_cuda LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)

find_package(BLAS REQUIRED)

add_executable(distance
    distance.cu
)

target_link_libraries(distance PRIVATE
    BLAS::BLAS
)

set(TILE_SIZE 16 CACHE STRING "CUDA tile size")
target_compile_definitions(distance PRIVATE TILE_SIZE=${TILE_SIZE})

target_compile_options(distance PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(distance PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)