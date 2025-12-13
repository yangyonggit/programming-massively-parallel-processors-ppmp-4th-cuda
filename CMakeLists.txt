cmake_minimum_required(VERSION 3.20)
project(PMPP_CUDA LANGUAGES CUDA CXX)

# Set CUDA runtime library to dynamic (required for iostream support)
set(CMAKE_CUDA_RUNTIME_LIBRARY DYNAMIC)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Keep MSVC default libs (required for iostream/std::cout)

# Find CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

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

# Chapter 03: RGB to Grayscale
add_executable(rgbToGray src/chapter03/rgb_to_gray.cu)
set_target_properties(rgbToGray PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/chapter03
)

# Link CUDA libraries
target_link_libraries(rgbToGray PRIVATE CUDA::cudart)

# Chapter 03: Blur Filter
add_executable(blurFilter src/chapter03/blur.cu)
set_target_properties(blurFilter PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/chapter03
)

target_link_libraries(blurFilter PRIVATE CUDA::cudart)

# Chapter 03: Matrix Multiplication (validate via cuBLAS)
add_executable(matrixMul src/chapter03/matrixMul.cu)
set_target_properties(matrixMul PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/chapter03
)
target_link_libraries(matrixMul PRIVATE CUDA::cudart CUDA::cublas)

# Chapter 03: Exercise 3.1 (user code)
add_executable(exercise_3_1 src/chapter03/exercise_3_1.cu)
set_target_properties(exercise_3_1 PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/chapter03
)
target_link_libraries(exercise_3_1 PRIVATE CUDA::cudart CUDA::cublas)

# Chapter 03: Exercise 3.2 (matrix-vector product)
add_executable(exercise_3_2 src/chapter03/exercise_3_2.cu)
set_target_properties(exercise_3_2 PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/chapter03
)
target_link_libraries(exercise_3_2 PRIVATE CUDA::cudart CUDA::cublas)

# Chapter 04: Query Device Info
add_executable(query_device src/chapter04/query_device.cu)
set_target_properties(query_device PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/chapter04
)
target_link_libraries(query_device PRIVATE CUDA::cudart)

# Copy assets directory to output location
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets")
    add_custom_command(TARGET rgbToGray POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/assets"
        "$<TARGET_FILE_DIR:rgbToGray>/../assets"
        COMMENT "Copying assets directory"
    )
    add_custom_command(TARGET blurFilter POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/assets"
        "$<TARGET_FILE_DIR:blurFilter>/../assets"
        COMMENT "Copying assets directory"
    )
    add_custom_command(TARGET matrixMul POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/assets"
        "$<TARGET_FILE_DIR:matrixMul>/../assets"
        COMMENT "Copying assets directory"
    )
    add_custom_command(TARGET exercise_3_1 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/assets"
        "$<TARGET_FILE_DIR:exercise_3_1>/../assets"
        COMMENT "Copying assets directory"
    )
    add_custom_command(TARGET exercise_3_2 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/assets"
        "$<TARGET_FILE_DIR:exercise_3_2>/../assets"
        COMMENT "Copying assets directory"
    )
    add_custom_command(TARGET query_device POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/assets"
        "$<TARGET_FILE_DIR:query_device>/../assets"
        COMMENT "Copying assets directory"
    )
endif()

# Print CUDA info
message(STATUS "CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "STB headers downloaded to: ${CMAKE_CURRENT_BINARY_DIR}/stb")
