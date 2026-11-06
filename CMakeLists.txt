cmake_minimum_required(VERSION 3.20)
project(kvclient LANGUAGES C CXX)

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)

add_subdirectory(./llama.cpp)

add_executable(inference ./inference/inference.cpp)

set_target_properties(inference PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")

target_link_libraries(inference PRIVATE llama)

target_include_directories(inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inference)