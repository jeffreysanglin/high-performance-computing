cmake_minimum_required(VERSION 3.20)

project(distance)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")

find_package(pybind11 REQUIRED)
find_package(BLAS REQUIRED)

add_library(distance_core STATIC
    source/distance_core.cpp
)
add_library(distance_core_serial STATIC
    source/distance_core_serial.cpp
)

target_include_directories(distance_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/source
)
target_include_directories(distance_core_serial PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/source
)

pybind11_add_module(distance_py ${CMAKE_CURRENT_SOURCE_DIR}/source/distance_bindings.cpp)

add_executable(distance_profile source/distance_main.cpp)
add_executable(distance_profile_serial source/distance_main_serial.cpp)

target_link_libraries(distance_core PRIVATE BLAS::BLAS)
target_link_libraries(distance_core_serial PRIVATE BLAS::BLAS)
target_link_libraries(distance_py PRIVATE distance_core)
target_link_libraries(distance_profile PRIVATE distance_core)
target_link_libraries(distance_profile_serial PRIVATE distance_core_serial)


set_target_properties(distance_py  PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)


