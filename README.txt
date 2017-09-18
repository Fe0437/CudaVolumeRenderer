# 🧊 Cuda Volume Renderer

Welcome to the **Cuda Volume Renderer** project! 🚀 This is a master's thesis project from 2017, focusing on high-performance volume rendering using CUDA. 

Initially designed to work with VTK MHD files, it now uses the more common OpenVDB format, thanks to a conversion tool that transforms MHD files into VDB. 📂

## 🌐 Learn More

For more details about the project, visit my [website](https://fe0437.github.io/thesis/) or read the [full thesis](Master_Thesis.pdf).

## 🛠️ Building the Project

### Prerequisites

Before you start, ensure you have the following installed:

- **CMake**: Required for building the project.
- **CUDA**: Tested with CUDA 12.0. Make sure you have a compatible version installed.
- **Docker**: Required for running the MHD to VDB conversion tool.

### Build Instructions

The project uses **vcpkg** to manage dependencies, which will be automatically downloaded if you use `build.sh`. Follow these simple steps to build the project:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/CudaVolumeRenderer.git
   cd CudaVolumeRenderer
   ```

2. **Run the Build Script:**
   ```bash
   ./build.sh
   ```

   You can also specify the compiler and build mode (Debug or Release) with the following commands:
   ```bash
   ./build.sh --debug -c clang
   ```

   Note: If no compiler is specified, `clang` is used by default as `gcc` is not available.

### Detailed Steps

- **Cleaning the Build Directory:** Use the `--clean` option to remove previous build files.
- **Compiler Selection:** Use `-c` or `--compiler` to specify `clang`.
- **Debug Mode:** Use `--debug` to build in Debug mode for development and testing.

## 🔄 Converting MHD to VDB

The project includes a tool to convert MHD files to VDB format using Docker. Follow these steps to perform the conversion:

1. **Navigate to the Conversion Directory:**
   ```bash
   cd scripts/convert-mhd
   ```

2. **Convert All MHD Files:**
   Run the following script to convert all MHD files in the `data/mhd` directory to VDB format:
   ```bash
   ./convert-all-mhd.sh
   ```

3. **Convert a Single MHD File:**
   To convert a specific MHD file, use the following command:
   ```bash
   ./convert-mhd.sh /path/to/input.mhd /path/to/output.vdb
   ```

   This will use Docker to run the conversion script inside a container, ensuring a consistent environment.

## 🖥️ Using the Command-Line Tool

After building the project, you can use the command-line tool to render scenes. Here's how to use it:

1. **Basic Usage:**
   ```bash
   ./CudaVolumeRenderer /path/to/scene.vdb
   ```
this is going to run an interactive renderer with the default parameters.
2. **Available Options:**

   - `--scene-file, -s`: Specify the scene file to parse.
   - `--scene-type`: Specify the scene type (`Auto`, `MitsubaXml`, `Vdb`, `Raw`). Defaults to `Auto`.
   - `--interactive`: Run the algorithm with the interactive view (default: true).
   - `--trials`: Number of times to run the algorithm (default: 1).
   - `--algorithm, -a`: Specify the algorithm to use (default: `cudaVolPath`).
   - `--kernel, -k`: Specify the kernel to use (default: `regenerationSK`).
   - `--number-of-tiles`: Specify the number of tiles (default: `1 1`).
   - `--use-unified-memory`: Use unified memory (default: false).
   - `--iterations, -i`: Number of iterations (default: 20).
   - `--output, -o`: Specify the output name.
   - `--resolution, -r`: Specify the resolution (default: `1024 1024`).

3. **Example Command:**
   ```bash
   ./CudaVolumeRenderer /path/to/scene.vdb --scene-type Vdb --iterations 50 --tiles 10 10
   ```

   This command will render the specified VDB scene with 50 iterations with 10x10 tiles.

## 📜 Project Overview

The Cuda Volume Renderer is designed to efficiently render large volumetric datasets using GPU acceleration. It leverages the power of CUDA for parallel computing, enabling the rendering of complex volumetric data with high performance. The project integrates several advanced libraries and tools:

- **OpenVDB**: A powerful library for volumetric data manipulation, providing efficient storage and access patterns.
- **CUDA**: Utilized for high-performance parallel computing, allowing the renderer to handle large datasets with ease.
- **Boost, Eigen, TBB**: These libraries offer additional computational and utility support, enhancing the renderer's capabilities.
- **GLEW and GLFW**: Used for managing OpenGL extensions and creating windows with OpenGL contexts, respectively.

## 🔮 Future Work

- **nanoVDB Integration**: Implementing nanoVDB for more efficient volume rendering.
- **User Interface**: Adding a graphical user interface for easier interaction.
- **NVIDIA OptiX Comparison**: Evaluating performance improvements with NVIDIA's OptiX.

## 📧 Contact

Thank you for your interest in the project! If you have any questions or suggestions, feel free to [contact me](mailto:federico.forti.1990@gmail.com). 😊

