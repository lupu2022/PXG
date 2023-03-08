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
 
list(APPEND CMAKE_CUDA_FLAGS "--threads 4")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
endif()
 
project(tensortype LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED cublas nvToolsExt)
find_package(Threads REQUIRED)

add_library(tensortype STATIC tensortype.cpp cuda_impl.cpp kernels/LtSgemm.cu)
target_include_directories(tensortype PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
