cmake_minimum_required(VERSION 3.10)

project(face_restore)

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

add_executable(convert convert.cpp)
target_link_libraries(convert nvinfer)
target_link_libraries(convert cudart)
target_link_libraries(convert nvonnxparser)

add_subdirectory(pybind11)
pybind11_add_module(FaceRestoration demo.cpp face_restoration.cpp)
target_link_libraries(FaceRestoration PRIVATE nvinfer)
target_link_libraries(FaceRestoration PRIVATE cudart)
target_link_libraries(FaceRestoration PRIVATE ${OpenCV_LIBS})
target_link_libraries(FaceRestoration PRIVATE pybind11::embed)
target_include_directories(FaceRestoration PUBLIC include)
set(SOURCES face_restoration.cpp face_restoration.hpp)
target_compile_definitions(FaceRestoration
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
