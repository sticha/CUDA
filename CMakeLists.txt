# CMakeLists.txt for the SuperResolution CUDA project

cmake_minimum_required(VERSION 2.8)
project(SuperResolution)

find_package(CUDA REQUIRED)

# Set options for NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch sm_35)
set(CUDA_SEPARABLE_COMPILATION ON)

# Find all sourcecode files
file(GLOB sources
    "./VisualProject/SuperResolution/*.cu"
    "./VisualProject/SuperResolution/*.h"
    )

# Compile sourcecode files
cuda_add_executable(SuperResolution ${sources})

# Link executable against OpenCV
target_link_libraries(
    SuperResolution
    -lopencv_highgui
    -lopencv_core
    )

