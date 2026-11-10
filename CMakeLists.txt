cmake_minimum_required(VERSION 3.20)
project(kvclient LANGUAGES C CXX)

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)

add_subdirectory(./llama.cpp)

target_sources(llama PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/kvc-inspect.cpp")
target_include_directories(llama PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/src" "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_library(kvclient STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kvclient.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kvc-chunk.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kvc-context.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kvc-storage.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/kvc-serialize.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/wrapper/llama-wrapper.cpp"
)
target_compile_features(kvclient PUBLIC cxx_std_17)

target_include_directories(kvclient PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include"
)

target_include_directories(kvclient PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

add_executable(inference ./inference/inference.cpp)

set_target_properties(inference PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")

target_link_libraries(inference PRIVATE llama kvclient)

target_include_directories(inference PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/inference"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

if(MINGW)
    target_link_options(inference PRIVATE -static-libgcc -static-libstdc++)
endif()