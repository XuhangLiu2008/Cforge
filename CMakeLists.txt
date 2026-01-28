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

# pybind
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bindings)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
add_subdirectory(third_party/pybind11)

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


# 清理构建目录
#rm -rf cmake-build-debug

# 重新配置
#mkdir -p cmake-build-debug && cd cmake-build-debug
#cmake .. -DCMAKE_PREFIX_PATH=/Users/xuhangliu/文件/编程/Cforge/third_party/libtorch