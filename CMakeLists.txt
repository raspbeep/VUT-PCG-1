#
# @file      CMakeLists.txt
#
# @author    David Bayer \n
#            Faculty of Information Technology \n
#            Brno University of Technology \n
#            ibayer@fit.vutbr.cz
#
# @brief     PCG Assignment 1
#
# @version   2024
#
# @date      04 October   2023, 09:00 (created) \n
#

cmake_minimum_required(VERSION 3.8)

set(CMAKE_CUDA_HOST_COMPILER         ${CMAKE_CXX_COMPILER})

project(Assignment1 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD               17)
set(CMAKE_CXX_STANDARD_REQUIRED      ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"))
  set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} -g")
  set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native -flto")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
  set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} /Od /Mtd")
  set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} /Ox /Mt /fp:fast /GL")
endif()

set(CMAKE_CUDA_STANDARD              17)
set(CMAKE_CUDA_STANDARD_REQUIRED     ON)
set(CMAKE_CUDA_ARCHITECTURES         61;70;75;80)
set(CMAKE_CUDA_FLAGS_DEBUG           "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
set(CMAKE_CUDA_FLAGS_RELEASE         "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -lineinfo")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  find_package(OpenMP)
endif()

include_directories(Commons)

# Gen
add_executable(gen Commons/gen.cpp)
target_link_libraries(gen HDF5::HDF5)

# h5Helper
add_library(h5Helper Commons/h5Helper.cpp)
target_include_directories(h5Helper PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(h5Helper PUBLIC HDF5::HDF5)

# Cpu
add_executable(nbodyCpu Cpu/main.cpp Cpu/nbody.cpp)
target_link_libraries(nbodyCpu h5Helper $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)

# Step0
add_executable(nbody0 Step0/main.cu Step0/nbody.cu)
target_link_libraries(nbody0 h5Helper CUDA::cudart)

# Step1
add_executable(nbody1 Step1/main.cu Step1/nbody.cu)
target_link_libraries(nbody1 h5Helper CUDA::cudart)

# Step2
add_executable(nbody2 Step2/main.cu Step2/nbody.cu)
target_link_libraries(nbody2 h5Helper CUDA::cudart)

# Step3
add_executable(nbody3 Step3/main.cu Step3/nbody.cu)
target_link_libraries(nbody3 h5Helper CUDA::cudart)

# Step4
add_executable(nbody4 Step4/main.cu Step4/nbody.cu)
target_link_libraries(nbody4 h5Helper CUDA::cudart)
