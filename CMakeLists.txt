cmake_minimum_required(VERSION 3.10)

project(face_restoration_trt)

add_definitions(-std=c++17)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(the_path_to/TensorRT/include)
link_directories(the_path_to/TensorRT/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(demo demo.cpp face_restoration.cpp)
target_link_libraries(demo nvinfer)
target_link_libraries(demo cudart)
target_link_libraries(demo ${OpenCV_LIBS})

add_executable(convert convert.cpp)
target_link_libraries(convert nvinfer)
target_link_libraries(convert cudart)
target_link_libraries(convert nvonnxparser)