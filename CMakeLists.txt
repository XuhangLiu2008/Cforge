cmake_minimum_required(VERSION 3.16)
project(Cforge)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE "Debug")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/share/cmake/Torch")


# torch
find_package(Torch REQUIRED
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/share/cmake/Torch" # where TorchConfigs.cmake is
)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/libtorch/include/torch/csrc/api/include")


# Cforge_lib
add_library(Cforge_lib
        csrc/filaMatch/Filaments.cpp
        csrc/filaMatch/BatchExpectPassMatrix.cpp
        csrc/_debugUtilis.cpp
        csrc/filaMatch/Optimizer.cpp
        third_party/json.hpp)
target_link_libraries(Cforge_lib "${TORCH_LIBRARIES}")


# executable
add_executable(Cforge
        csrc/main.cpp
        csrc/test.cpp)

target_link_libraries(Cforge Cforge_lib)