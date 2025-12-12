cmake_minimum_required(VERSION 3.20)
project(PMPP_CUDA LANGUAGES CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Disable unnecessary default libraries for Windows
if(MSVC)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:msvcprt")
endif()

# Find CUDA
find_package(CUDA REQUIRED)

# Create a module to download external dependencies
include(cmake/DownloadSTB.cmake)

# Download STB library
download_stb_headers()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/stb)

# Chapter 02: Vector Addition
add_executable(vectorAdd src/chapter02/vectorAdd.cu)
set_target_properties(vectorAdd PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/chapter02
)

# Explicitly link only required CUDA libraries
target_link_libraries(vectorAdd PRIVATE cuda cudart)

# Print CUDA info
message(STATUS "CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "STB headers downloaded to: ${CMAKE_CURRENT_BINARY_DIR}/stb")
