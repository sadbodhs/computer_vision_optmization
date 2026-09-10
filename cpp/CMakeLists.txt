cmake_minimum_required(VERSION 3.18)
# Builds the five Triton/TensorRT benchmark binaries. Names match README §10/§12:
#   trt_pipeline       (A1) src/main.cpp            — C++ TRT, CPU path
#   trt_pipeline_cuda  (A2) src/main_cuda.cu        — C++ TRT, full-CUDA zero-copy
#   trt_grpc_client    (B1) src/grpc_client.cpp     — Triton + C++ client, raw gRPC
#   trt_grpc_cuda      (B2) src/grpc_client_cuda.cu — Triton + C++ client, CUDA shm (also D-sync)
#   trt_grpc_async     (D)  src/grpc_async_client.cu— Triton async, 8 in-flight, dyn batch
# ds_bench (E, DeepStream) is NOT built here — it needs GStreamer + DeepStream headers
# that only exist in the ds-build image; see docker/build_ds_bench.sh.
project(trt_pipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale)

# TensorRT (headers/libs ship in the Triton container; override if elsewhere).
find_library(NVINFER nvinfer)
find_library(NVINFER_PLUGIN nvinfer_plugin)

# Triton C++ client SDK (grpc_client.h + libgrpcclient). In the tritonserver *-sdk
# image it lives under /workspace/install; in triton-bench:v3 it is under /opt.
# Override with -DTRITON_CLIENT_ROOT=/path if your install differs.
set(TRITON_CLIENT_ROOT "/workspace/install" CACHE PATH "Triton C++ client SDK root")
find_path(TRITON_CLIENT_INC grpc_client.h
  HINTS ${TRITON_CLIENT_ROOT}/include /opt/tritonclient/include /opt/include)
find_library(TRITON_GRPC_LIB grpcclient
  HINTS ${TRITON_CLIENT_ROOT}/lib /opt/tritonclient/lib /opt/lib)

# ---- A1: C++ TRT, CPU path -------------------------------------------------
add_executable(trt_pipeline src/main.cpp)
target_include_directories(trt_pipeline PRIVATE ${FFMPEG_INCLUDE_DIRS})
target_link_directories(trt_pipeline PRIVATE ${FFMPEG_LIBRARY_DIRS})
target_link_libraries(trt_pipeline PRIVATE ${FFMPEG_LIBRARIES} ${NVINFER} ${NVINFER_PLUGIN} CUDA::cudart)

# ---- A2: C++ TRT, full-CUDA zero-copy --------------------------------------
add_executable(trt_pipeline_cuda src/main_cuda.cu)
target_include_directories(trt_pipeline_cuda PRIVATE ${FFMPEG_INCLUDE_DIRS})
target_link_directories(trt_pipeline_cuda PRIVATE ${FFMPEG_LIBRARY_DIRS})
target_link_libraries(trt_pipeline_cuda PRIVATE ${FFMPEG_LIBRARIES} ${NVINFER} ${NVINFER_PLUGIN} CUDA::cudart)

# ---- Triton client arms (need the C++ client SDK) --------------------------
if(TRITON_CLIENT_INC AND TRITON_GRPC_LIB)
  # B1: raw gRPC
  add_executable(trt_grpc_client src/grpc_client.cpp)
  target_include_directories(trt_grpc_client PRIVATE ${TRITON_CLIENT_INC} ${FFMPEG_INCLUDE_DIRS})
  target_link_directories(trt_grpc_client PRIVATE ${FFMPEG_LIBRARY_DIRS})
  target_link_libraries(trt_grpc_client PRIVATE ${TRITON_GRPC_LIB} ${FFMPEG_LIBRARIES} CUDA::cudart)

  # B2 / D-sync: CUDA shared memory
  add_executable(trt_grpc_cuda src/grpc_client_cuda.cu)
  target_include_directories(trt_grpc_cuda PRIVATE ${TRITON_CLIENT_INC} ${FFMPEG_INCLUDE_DIRS})
  target_link_directories(trt_grpc_cuda PRIVATE ${FFMPEG_LIBRARY_DIRS})
  target_link_libraries(trt_grpc_cuda PRIVATE ${TRITON_GRPC_LIB} ${FFMPEG_LIBRARIES} CUDA::cudart)

  # D: async, 8 in-flight, dynamic batching
  add_executable(trt_grpc_async src/grpc_async_client.cu)
  target_include_directories(trt_grpc_async PRIVATE ${TRITON_CLIENT_INC})
  target_link_libraries(trt_grpc_async PRIVATE ${TRITON_GRPC_LIB} CUDA::cudart)
else()
  message(WARNING "Triton C++ client SDK not found (grpc_client.h / libgrpcclient). "
                  "Skipping trt_grpc_client / trt_grpc_cuda / trt_grpc_async. "
                  "Set -DTRITON_CLIENT_ROOT=/path to your SDK install.")
endif()
