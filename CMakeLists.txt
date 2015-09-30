# CMakeLists.txt for the SuperResolution CUDA project

cmake_minimum_required(VERSION 3.0)
project(SuperResolution)

find_package(CUDA REQUIRED)

# Set options for NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch sm_35;-std=c++11)
set(CUDA_SEPARABLE_COMPILATION ON)

# Find all sourcecode files
file(GLOB sources
    "./VisualProject/SuperResolution/*.cu"
    "./VisualProject/SuperResolution/*.h"
    )
file(GLOB test_sources "./VisualProject/SuperResolution/*Tests.cu")
file(GLOB normal_main "./VisualProject/SuperResolution/*ain.cu")
set(test_build ${sources})
list(REMOVE_ITEM sources ${test_sources})
list(REMOVE_ITEM test_build ${normal_main})

# Compile sourcecode files
cuda_add_executable(SuperResolution ${sources})
cuda_add_executable(runTests ${test_build})

# Link executable against OpenCV
target_link_libraries(
    SuperResolution
    -lopencv_highgui
    -lopencv_core
    )

target_link_libraries(
    runTests
    -lopencv_highgui
    -lopencv_core
    )
