cmake_minimum_required(VERSION 3.20)

project(distance LANGUAGES CXX)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -fopenmp")

find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)
find_package(pybind11 REQUIRED)


pybind11_add_module(distance_py ${CMAKE_CURRENT_SOURCE_DIR}/source/distance_bindings.cpp)

add_executable(distance_profile source/distance_main.cpp)

target_link_libraries(distance_py PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    BLAS::BLAS
)

target_link_libraries(distance_profile PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    BLAS::BLAS
)

set_target_properties(distance_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
