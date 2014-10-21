# The name of the project.
project(Crixus)

# release or debug mode
set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE "Debug")

# build path
set(EXECUTABLE_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})
set(ENV{HDF5_ROOT} "/home/arnom/software/shdf5")

# CMake Minimum verion required. 
# FindCUDA script is distributed since version 2.8
cmake_minimum_required(VERSION 2.8)

# Search packages.
find_package(CUDA REQUIRED)
find_package(HDF5 REQUIRED)

# set compilation options
set(CUDA_NVCC_FLAGS "-arch=sm_20")

include_directories (${CUDA_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS}) 

# Add source directory to project.
add_subdirectory(src)

#Always compile Crixus
add_executable(
  Crixus
  src/crixus.cpp
)

target_link_libraries(
  Crixus Core ${HDF5_LIBRARIES} cuCore
)
