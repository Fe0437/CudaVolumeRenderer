cmake_minimum_required(VERSION 3.18)

project(CudaVolumeRenderer LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Add external libraries
set(STB_ROOT ${CMAKE_SOURCE_DIR}/external/stb)
set(GLM_ROOT ${CMAKE_SOURCE_DIR}/external/glm)
set(GLFW_ROOT ${CMAKE_SOURCE_DIR}/external/glfw)
set(HELPER_CUDA_ROOT ${CMAKE_SOURCE_DIR}/external/helper_cuda)
set(GLEW_ROOT ${CMAKE_SOURCE_DIR}/external/glew)
set(BOOST_ROOT ${CMAKE_SOURCE_DIR}/external/boost)
find_package(Boost REQUIRED)
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

# Add executable
add_executable(CudaVolumeRenderer ${SOURCES})

# Include directories for external libraries
target_include_directories(CudaVolumeRenderer PRIVATE 
    ${CUDA_INCLUDE_DIRS} 
    ${Boost_INCLUDE_DIRS}
    ${STB_ROOT}
    ${GLM_ROOT}
    ${GLFW_ROOT}/include
    ${HELPER_CUDA_ROOT}
    ${GLEW_ROOT}/include
)
target_include_directories(CudaVolumeRenderer PRIVATE ${CUDA_INCLUDE_DIRS})

# Link CUDA libraries
target_link_libraries(CudaVolumeRenderer ${CUDA_LIBRARIES})

# Set CUDA properties
set_target_properties(CudaVolumeRenderer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
)
