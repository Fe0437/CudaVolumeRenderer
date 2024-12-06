#!/bin/bash

# Create a build directory if it doesn't exist
mkdir -p build
cd build

# Run CMake to configure the project
cmake ..

# Build the project
cmake --build .

# Return to the original directory
cd ..
