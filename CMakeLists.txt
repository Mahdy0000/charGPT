cmake_minimum_required(VERSION 3.18)
project(llm_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# RTX 3050 = SM86 (Ampere)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Optimization flags
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /fp:fast /EHsc /DNDEBUG")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")
endif()

# Training executable
add_executable(llm src/main.cpp src/cuda_kernels.cu)
target_link_libraries(llm PRIVATE cublas)
set_target_properties(llm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Chat executable
add_executable(chat src/chat.cpp src/cuda_kernels.cu)
target_link_libraries(chat PRIVATE cublas)
set_target_properties(chat PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Static link C++ runtime on MinGW
if(MINGW)
    target_link_options(llm PRIVATE -static)
    target_link_options(chat PRIVATE -static)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
