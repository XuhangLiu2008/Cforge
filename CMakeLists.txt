cmake_minimum_required(VERSION 3.16)
project(Cforge)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Torch REQUIRED)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/libtorch")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/libtorch/include/torch/csrc/api/include")

add_library(Cforge_lib
        Filaments.cpp
        BatchExpectPassMatrix.cpp
        _debugUtilis.cpp
        Optimizer.cpp
        Bitmap.cpp
        photoProcess.cpp
        generate3D.cpp
        json.hpp)

target_link_libraries(Cforge_lib "${TORCH_LIBRARIES}")

add_executable(Cforge
        main.cpp
        test.cpp)

target_link_libraries(Cforge Cforge_lib)