#!/bin/bash

set -o pipefail  # Ensure errors propagate in pipes

# Function to handle errors
handle_error() {
    echo "Build failed!"
    exit 1
}

# Set trap for error handling
trap 'handle_error' ERR

# Default values
BUILD_TYPE="Release"
COMPILER=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            echo "Cleaning the build directory..."
            rm -rf build
            shift
            ;;
        -c|--compiler)
            COMPILER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--debug] [--clean] [-c|--compiler <gcc|clang>]"
            exit 1
            ;;
    esac
done

# Install dependencies with vcpkg
echo "Installing dependencies with vcpkg..."
if [ ! -d "vcpkg" ]; then
    echo "Cloning and bootstrapping vcpkg..."
    git clone https://github.com/microsoft/vcpkg.git || handle_error
    ./vcpkg/bootstrap-vcpkg.sh || handle_error
fi

# vcpkg will automatically use vcpkg.json to install dependencies

# Build the main project with specified configuration
echo "Building the main project in ${BUILD_TYPE} mode..."
mkdir -p build || handle_error
cd build || handle_error

# Configure compiler flags
CMAKE_FLAGS="-DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE"

if [ "$COMPILER" = "clang" ]; then
    export CC=clang
    export CXX=clang++
    CMAKE_FLAGS="$CMAKE_FLAGS -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++"
fi

# Run CMake
cmake $CMAKE_FLAGS ../implementation || handle_error

# Build
cmake --build . --config $BUILD_TYPE -j $(nproc) || handle_error

# Install only in Release mode
if [ "$BUILD_TYPE" = "Release" ]; then
    echo "Installing in Release mode..."
    cmake --install . --config Release || handle_error
fi

echo "Build completed successfully in ${BUILD_TYPE} mode!"
