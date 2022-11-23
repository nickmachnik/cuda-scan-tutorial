cmake_minimum_required(VERSION 3.18)

project(MPS VERSION 1.0
            DESCRIPTION "Example code showcasing parallel computation of sums"
            LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

# add this for testing
# if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
#     include(CTest)
# endif()

# if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
#     enable_testing()
#     include(GoogleTest)
#     add_subdirectory(tests)
# endif()

add_subdirectory(src)
add_subdirectory(bin)
