cmake_minimum_required(VERSION 3.21)

project(CudaVolumeRenderer VERSION 1.0 LANGUAGES CXX CUDA)

# Standard settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

# Enable CUDA debug information
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g --diag-suppress 20012")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g --diag-suppress 20012")
endif()

# Download CPM.cmake if not already present
set(CPM_DOWNLOAD_VERSION 0.40.2)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
    message(STATUS "Downloading CPM.cmake...")
    file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()
include(${CPM_DOWNLOAD_LOCATION})


set(Boost_USE_STATIC_LIBS        ON)  # only find static libs
set(Boost_USE_DEBUG_LIBS        OFF)  # ignore debug libs and
set(Boost_USE_RELEASE_LIBS       ON)  # only find release libs
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME     ON)  # link Boost Static libraries
find_package(Boost REQUIRED COMPONENTS program_options)

# Major dependencies (handled by vcpkg)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenVDB CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(GLEW REQUIRED)
add_definitions(-DGLEW_DLL)
find_package(glfw3 CONFIG REQUIRED)


# Lightweight dependencies (handled by CPM)
CPMAddPackage(NAME glm GITHUB_REPOSITORY g-truc/glm GIT_TAG 1.0.1)
add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)
CPMAddPackage(NAME pugixml GITHUB_REPOSITORY zeux/pugixml VERSION 1.14 OPTIONS "PUGIXML_BUILD_TESTS OFF")
CPMAddPackage(NAME stb GITHUB_REPOSITORY nothings/stb GIT_TAG master DOWNLOAD_ONLY YES)

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Collect all source files
file(GLOB_RECURSE CPP_SOURCES 
    "src/*.cpp"
    "src/*.hpp"
    "src/*.h"
)
file(GLOB_RECURSE CUDA_SOURCES 
    "src/*.cu"
    "src/*.cuh"
)

# Create a library for OpenVDB
add_library(openvdb_lib STATIC
    vdb_adapter/VDBAdapter.cpp
    vdb_adapter/VDBAdapter.h
)

# Include directories for OpenVDB library
target_include_directories(openvdb_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vdb_adapter
    ${glm_SOURCE_DIR}
    ${Eigen3_INCLUDE_DIRS}
)

# Link libraries for OpenVDB library
target_link_libraries(openvdb_lib PRIVATE
    OpenVDB::openvdb
    TBB::tbb
    glm
)

# Add main executable
add_executable(${PROJECT_NAME} 
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
)

# Set target properties for CUDA
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON  # Help with device symbol resolution
)

# Add CUDA debug options to the target
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -G
            -g
            --source-in-ptx
            --diag-suppress 20012
        >
    )
endif()

# Include directories for main executable
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${glfw_SOURCE_DIR}/include
    ${glm_SOURCE_DIR}
    ${stb_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/vdb_adapter
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link libraries for main executable
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${CUDAToolkit_LIBRARIES}
    GLEW::GLEW
    glfw
    glm
    pugixml
    Eigen3::Eigen
    openvdb_lib
    TBB::tbb
    Boost::program_options
)

# Installation
install(TARGETS ${PROJECT_NAME} 
    RUNTIME DESTINATION bin
)

# Install runtime dependencies
install(DIRECTORY 
    ${CMAKE_BINARY_DIR}/$<CONFIG>/
    DESTINATION bin
    FILES_MATCHING 
    PATTERN "*.dll"
    PATTERN "*.so*"
    PATTERN "*.dylib"
)

