cmake_minimum_required(VERSION 3.16)
project(gguf_train LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp" CACHE PATH "Path to vendored llama.cpp")

option(GGML_CUDA   "Enable CUDA backend in llama.cpp/ggml" OFF)
option(GGML_VULKAN "Enable Vulkan backend in llama.cpp/ggml" OFF)
option(GGML_METAL  "Enable Metal backend in llama.cpp/ggml" OFF)
option(GGML_HIP    "Enable ROCm/HIP backend in llama.cpp/ggml" OFF)
option(GGML_SYCL   "Enable SYCL backend in llama.cpp/ggml" OFF)
option(GGML_OPENCL "Enable OpenCL backend in llama.cpp/ggml" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(LLAMA_BUILD_COMMON "Build llama common" OFF)
option(LLAMA_BUILD_TESTS "Build llama tests" OFF)
option(LLAMA_BUILD_EXAMPLES "Build llama examples" OFF)
option(LLAMA_BUILD_TOOLS "Build llama tools" OFF)
option(LLAMA_CURL "Enable curl" OFF)

add_subdirectory("${LLAMA_CPP_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama_cpp" EXCLUDE_FROM_ALL)

find_package(OpenMP)

add_library(gguf_train SHARED
    gt_core.cpp
    gt_ops_math.cpp
    gt_ops_nn.cpp
    gt_ops_extra.cpp
)
target_include_directories(gguf_train PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${LLAMA_CPP_DIR}/ggml/include"
)
# GT_BUILD_SHARED is the macro gguf_train.h keys GT_API on (dllexport when building the DLLs).
target_compile_definitions(gguf_train PRIVATE GT_BUILD_SHARED GGML_SHARED)
target_link_libraries(gguf_train PRIVATE ggml)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gguf_train PRIVATE OpenMP::OpenMP_CXX)
endif()

add_library(llama_bridge SHARED
    llama_bridge.cpp
)
target_include_directories(llama_bridge PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include"
)
target_compile_definitions(llama_bridge PRIVATE GT_BUILD_SHARED GGML_SHARED LLAMA_SHARED)
target_link_libraries(llama_bridge PRIVATE gguf_train llama ggml)

if(APPLE)
    set_target_properties(gguf_train llama_bridge PROPERTIES
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@loader_path"
    )
elseif(UNIX)
    set_target_properties(gguf_train llama_bridge PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
endif()
