cmake_minimum_required(VERSION 3.21)
set(MYPROJ CudaVolumeRenderer)
project(${MYPROJ} VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Download CPM.cmake
set(CPM_DOWNLOAD_VERSION 0.40.2)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake...")
    file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()
include(${CPM_DOWNLOAD_LOCATION})

# Include GLFW
CPMAddPackage(
  NAME glfw
  GITHUB_REPOSITORY glfw/glfw
  GIT_TAG 3.4
  OPTIONS
    "GLFW_BUILD_DOCS OFF"
    "GLFW_BUILD_EXAMPLES OFF"
    "GLFW_BUILD_TESTS OFF"
    "GLFW_BUILD_WAYLAND OFF"
    "GLFW_INSTALL OFF"
)

# Include GLM
CPMAddPackage(
  NAME glm
  GITHUB_REPOSITORY g-truc/glm
  GIT_TAG 1.0.1
)

# Include argparse
CPMAddPackage(
  NAME argparse
  GITHUB_REPOSITORY p-ranav/argparse
  VERSION 2.1
)

# Include VTK
CPMAddPackage(
    NAME VTK
    GITHUB_REPOSITORY Kitware/VTK
    VERSION 8.0.1
    OPTIONS
        "VTK_BUILD_ALL_MODULES OFF"
        "VTK_BUILD_EXAMPLES OFF"
        "VTK_BUILD_TESTING OFF"
        "BUILD_SHARED_LIBS ON"
)

# Include CUB
CPMAddPackage(
    NAME CUB
    GITHUB_REPOSITORY NVIDIA/cub
    GIT_TAG 1.16.0
)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Add sources
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

# Add executable
add_executable(${MYPROJ} ${SOURCES})

# Include directories for external libraries
target_include_directories(${MYPROJ} PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
    ${glfw_SOURCE_DIR}/include
    ${glm_SOURCE_DIR}
    ${argparse_SOURCE_DIR}/include
    ${CUB_SOURCE_DIR}
    ${VTK_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${MYPROJ} PRIVATE 
    ${CUDAToolkit_LIBRARIES}  # Link CUDA libraries
    glfw
    glm
    argparse
    vtkCommonCore
)

# Compile definitions
add_definitions(-DGLM_ENABLE_EXPERIMENTAL)

# CUDA-specific properties
set_target_properties(${MYPROJ} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "52;60;61;70;75;80;86"
)

# Copy .jpg image next to the executable
file(GLOB DATA "${CMAKE_SOURCE_DIR}/data")
add_custom_command(
  TARGET ${MYPROJ} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${DATA} $<TARGET_FILE_DIR:${MYPROJ}>
)

# Set a default installation prefix
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/_install" CACHE PATH "Default install path" FORCE)

# Define the installation directory for the executable
install(TARGETS ${MYPROJ}
    RUNTIME DESTINATION bin
)

# Install images
install(FILES ${IMAGE_FILE}
    DESTINATION bin
)
