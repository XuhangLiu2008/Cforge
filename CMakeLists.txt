cmake_minimum_required(VERSION 3.16)
project(Cforge)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE "Debug")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

# torch
set(TORCH_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/third_party/libtorch")
set(Torch_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch/share/cmake/Torch)

find_package(Torch REQUIRED)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch/include/torch/csrc/api/include")


message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")
message(STATUS "TORCH_LIBRARY_DIR: ${TORCH_LIBRARY_DIR}")

# Cforge_lib
add_library(Cforge_lib
        csrc/filaMatch/Filaments.cpp
        csrc/filaMatch/BatchExpectPassMatrix.cpp
        csrc/_debugUtilis.cpp
        csrc/filaMatch/Optimizer.cpp
        third_party/json.hpp)
target_link_libraries(Cforge_lib ${TORCH_LIBRARIES})


# executable
add_executable(Cforge
        csrc/main.cpp
        csrc/test.cpp)

target_link_libraries(Cforge Cforge_lib)