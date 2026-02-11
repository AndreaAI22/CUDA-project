cmake_minimum_required(VERSION 3.18)

set(CMAKE_C_COMPILER /usr/bin/gcc-10)
set(CMAKE_CXX_COMPILER /usr/bin/g++-10)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-10)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-10)

project(cuda_project LANGUAGES CXX CUDA)

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

add_executable(app
  src/video_pipeline.cpp
  src/optical_flow.cu
  src/crop.cu
  src/edge_detection_sobel.cu
  src/scale.cu
  src/rotate.cu
)

target_link_libraries(app PRIVATE ${OpenCV_LIBS} cuda)
target_include_directories(app PRIVATE
   ${OpenCV_INCLUDE_DIRS}
   ${CMAKE_SOURCE_DIR}/include
   )

set_target_properties(app PROPERTIES
  CXX_STANDARD 17
  CUDA_STANDARD 14
)
