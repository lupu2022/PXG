# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)
 
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 )
endif()
 
 
set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CUDA_STANDARD 17) 
set(CMAKE_CUDA_STANDARD_REQUIRED ON) 
 
project(transformer_engine LANGUAGES CUDA CXX)
 
list(APPEND CMAKE_CUDA_FLAGS "--threads 4")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
endif()
 
find_package(CUDAToolkit REQUIRED cublas nvToolsExt)

project(kernels LANGUAGES CUDA CXX)

set(cuda_kernel_files
    kernels/util.cc.cu
    kernels/cross_entropy.cu
    kernels/cublas_wrappers.cu
    kernels/cuda_util.cu
    kernels/dropout_kernels.cu
    kernels/embedding_kernels.cu
    kernels/embKernels.cc.cu
    kernels/# fused_adam_kernel.cu
    kernels/general_kernels.cu
    kernels/gptKernels.cc.cu
    kernels/normalize_kernels.cu
    kernels/softmax_kernels.cu
    kernels/softmax_kernels_new.cu
    kernels/transform_kernels.cu
    kernels/transform_kernels_new.cu
    kernels/crf.cu
    kernels/transformerKernels.cc.cu)

add_library(ls-kernels STATIC ${cuda_kernel_files})
target_include_directories(ls-kernels PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/includes")

add_library(ls-layers STATIC layers.cpp)
target_include_directories(ls-layers PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/includes")
target_include_directories(ls-layers PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

