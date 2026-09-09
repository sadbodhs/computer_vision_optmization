cmake_minimum_required(VERSION 3.16)
project(trt_pipeline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale)

add_executable(trt_pipeline src/main.cpp)

target_include_directories(trt_pipeline PRIVATE
  ${FFMPEG_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  /usr/include/x86_64-linux-gnu)

target_link_directories(trt_pipeline PRIVATE
  ${FFMPEG_LIBRARY_DIRS}
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

target_link_libraries(trt_pipeline PRIVATE
  ${FFMPEG_LIBRARIES}
  nvinfer nvinfer_plugin
  cudart)
