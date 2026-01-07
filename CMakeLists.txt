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

# Chapter 05: Tiled Matrix Multiplication
add_executable(tiled_matrix_multiplication src/chapter05/tiled_matrix_multiplication.cu)
set_target_properties(tiled_matrix_multiplication PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/chapter05
)
target_link_libraries(tiled_matrix_multiplication PRIVATE CUDA::cudart CUDA::cublas)

# Chapter 06: Thread Coarsening Matrix Multiplication
add_executable(thread_coarsening_matrix_mul src/chapter06/thread_coarsening_matrix_mul.cu)
set_target_properties(thread_coarsening_matrix_mul PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/chapter06
)
target_link_libraries(thread_coarsening_matrix_mul PRIVATE CUDA::cudart CUDA::cublas)

# Chapter 06: Exercise 6.1 (Corner Turning)
add_executable(exercise_6_1 src/chapter06/exercise_6_1.cu)
set_target_properties(exercise_6_1 PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/chapter06
)
target_link_libraries(exercise_6_1 PRIVATE CUDA::cudart CUDA::cublas)

# Chapter 07: 2D Convolution with multiple kernels and filters
add_executable(convolution src/chapter07/convolution.cu)
set_target_properties(convolution PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/chapter07
)
target_link_libraries(convolution PRIVATE CUDA::cudart)

# Chapter 07: Exercise 7.8 (3D Convolution)
add_executable(exercise_7_8 src/chapter07/exercise_7_8.cu)
set_target_properties(exercise_7_8 PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/chapter07
)
target_link_libraries(exercise_7_8 PRIVATE CUDA::cudart)

# Chapter 07: Exercise 7.9 (3D Convolution - constant memory)
add_executable(exercise_7_9 src/chapter07/exercise_7_9.cu)
set_target_properties(exercise_7_9 PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/chapter07
)
target_link_libraries(exercise_7_9 PRIVATE CUDA::cudart)

# Chapter 07: Exercise 7.10 (3D Tiled Convolution - constant memory)
add_executable(exercise_7_10 src/chapter07/exercise_7_10.cu)
set_target_properties(exercise_7_10 PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/chapter07
)
target_link_libraries(exercise_7_10 PRIVATE CUDA::cudart)

# Chapter 10: Sum Reduction
add_executable(sum_reduction src/chapter10/sum_reduction.cu)
set_target_properties(sum_reduction PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/chapter10
)
target_link_libraries(sum_reduction PRIVATE CUDA::cudart)

# Chapter 10: Exercise 10.3 (Convergent reduction variant)
add_executable(excercise_10_3 src/chapter10/excercise_10_3.cu)
set_target_properties(excercise_10_3 PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/chapter10
)
target_link_libraries(excercise_10_3 PRIVATE CUDA::cudart)

# Chapter 10: Exercise 10.4&5 (Max reduction)
add_executable(excercise_10_4_5 src/chapter10/excercise_10_4_5.cu)
set_target_properties(excercise_10_4_5 PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/chapter10
)
target_link_libraries(excercise_10_4_5 PRIVATE CUDA::cudart)

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
    add_custom_command(TARGET tiled_matrix_multiplication POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/assets"
        "$<TARGET_FILE_DIR:tiled_matrix_multiplication>/../assets"
        COMMENT "Copying assets directory"
    )
    add_custom_command(TARGET thread_coarsening_matrix_mul POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/assets"
        "$<TARGET_FILE_DIR:thread_coarsening_matrix_mul>/../assets"
        COMMENT "Copying assets directory"
    )
    add_custom_command(TARGET exercise_6_1 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/assets"
        "$<TARGET_FILE_DIR:exercise_6_1>/../assets"
        COMMENT "Copying assets directory"
    )
    add_custom_command(TARGET convolution POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/assets"
        "$<TARGET_FILE_DIR:convolution>/../assets"
        COMMENT "Copying assets directory"
    )
endif()

# Print CUDA info
message(STATUS "CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "STB headers downloaded to: ${CMAKE_CURRENT_BINARY_DIR}/stb")
