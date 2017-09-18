#ifndef CONFIG_H_
#define CONFIG_H_

// 1. CUDA headers first
#include <cuda_runtime.h>

#include "helper_cuda.h"

// 2. System/Standard headers
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// 3. Project headers
#include "Image.h"
#include "Scene.h"

/*
how to use

if (vm.count("help")) {
        cout << desc << "\n";
        return 0;
}
        cout << "Compression level was set to "
                 << vm["compression"].as<double>() << ".\n";
} else {
        cout << "Compression level was not set.\n";
}

*/

struct CudaConfig {
  // CUDA device properties
  int devId = -1;
  cudaDeviceProp device_properties{};
  bool unified_memory = false;

  // Kernel configuration
  dim3 block_size = dim3(32);
  dim3 grid_size = dim3(1);
  size_t dynamic_shared_memory = 0;

  // Occupancy information
  int max_active_blocks_per_sm = 0;
  int max_total_blocks = 0;
};

struct PathTracingConfig {
  // PATH TRACING
  float max_time{-1};
  unsigned int iterations;
  unsigned int max_path_length{1000};

  explicit PathTracingConfig(unsigned int _iterations = 10)
      : iterations(_iterations) {}
};

struct TilingConfig {
  // PATH TRACING
  uint2 n_tiles{};
  uint2 tile_dim{};
  uint2 resolution{};

  explicit TilingConfig(uint2 _resolution = make_uint2(400, 400),
                        uint2 n_tiles = make_uint2(1, 1))
      : resolution(_resolution), n_tiles(n_tiles) {
    tile_dim.x = (int)ceil(resolution.x / n_tiles.x);
    tile_dim.y = (int)ceil(resolution.y / n_tiles.y);
  }

  ~TilingConfig() = default;
  TilingConfig(const TilingConfig& other) = default;
  TilingConfig(TilingConfig&& other) noexcept = default;
  auto operator=(const TilingConfig& other) -> TilingConfig& = default;
};

class Config {
 public:
  enum Algorithm {
    kCudaVolPath,
    kAlgorithmUnknown,
  };

  enum Kernel {
    kNaivesk,
    kNaivemk,
    kRegenerationsk,
    kStreamingmk,
    kStreamingsk,
    kSortingsk,
    kKernelUnknown
  };

  Kernel kernel = kNaivesk;
  Algorithm algorithm = kCudaVolPath;

  int test_trials = 1;
  uint2 resolution{400, 400};
  std::string output_name = "test";
  bool interactive = false;
  Scene scene;

  PathTracingConfig path_tracing_config;
  TilingConfig tiling_config;
  CudaConfig cuda_config;

  Config() = default;

  explicit Config(const Scene& _scene,
                  Algorithm _algorithm = Algorithm::kCudaVolPath)
      : scene(_scene), algorithm(_algorithm) {
    auto res = scene.getCamera()->getResolution();
    resolution = {(uint)res.x, (uint)res.y};
  }

  auto cudaConfig(CudaConfig _cuda_config) -> Config& {
    cuda_config = _cuda_config;

    if (MAX_THREADS_PER_BLOCK >
        cuda_config.device_properties.maxThreadsPerBlock) {
      LOG_CONFIG(
          "WARNING: MAX_THREADS_PER_BLOCK (%d) exceeds device maximum (%d)\n",
          MAX_THREADS_PER_BLOCK,
          cuda_config.device_properties.maxThreadsPerBlock);
    }

    if (STREAMING_THREADS_BLOCK >
        cuda_config.device_properties.maxThreadsPerBlock) {
      LOG_CONFIG(
          "WARNING: STREAMING_THREADS_BLOCK (%d) exceeds device maximum (%d)\n",
          STREAMING_THREADS_BLOCK,
          cuda_config.device_properties.maxThreadsPerBlock);
    }

    // Check shared memory requirements
    size_t sharedMemPerBlock = cuda_config.device_properties.sharedMemPerBlock;
    if (STREAMING_SHARED_MEMORY > sharedMemPerBlock) {
      LOG_CONFIG(
          "WARNING: STREAMING_SHARED_MEMORY (%d) exceeds device shared memory "
          "per block (%zu)\n",
          STREAMING_SHARED_MEMORY, sharedMemPerBlock);
    }

    // Check memory requirements for volume data
    auto albedo = scene.getMedium().albedo_volume;
    size_t albedo_texture_bytes = albedo.getBytes();
    float max_memory_threshold = 0.8;

    if (cuda_config.device_properties.totalGlobalMem * max_memory_threshold <
        albedo_texture_bytes) {
      cuda_config.unified_memory = true;
      COUT_DEBUG("not enough device global memory -> using unified memory")
    }

    return *this;
  }

  auto kernelConfig(Kernel _kernel) -> Config& {
    kernel = _kernel;
    return *this;
  }

  auto algorithmConfig(Algorithm _alg) -> Config& {
    algorithm = _alg;
    return *this;
  }

  auto pathTracingConfig(PathTracingConfig _path_tracing_config) -> Config& {
    path_tracing_config = _path_tracing_config;
    return *this;
  }

  auto tilingConfig(TilingConfig _tiling_config) -> Config& {
    tiling_config = _tiling_config;
    assert(tiling_config.resolution.x == resolution.x);
    assert(tiling_config.resolution.y == resolution.y);
    return *this;
  }

  static std::vector<std::string> getAlgorithmNamesOrderedVector() {
    return {"cudaVolPath"};
  }

  static auto getAlgorithmAcronym(Algorithm alg) -> const std::string {
    auto algs = getAlgorithmNamesOrderedVector();
    assert(
        algs.size() == Algorithm::kAlgorithmUnknown &&
        "check the ConfigParser class algorithms, kUnknown must be at the end");
    if (alg > algs.size()) {
      LOG_CONFIG(" algorithm not present in the list of algorithms");
    }
    if (alg < 0) {
      LOG_CONFIG(" error value < 0");
    }
    return algs[alg];
  }

  static auto getAlgorithm(std::string alg) -> Algorithm {
    for (int i = 0; i < Algorithm::kAlgorithmUnknown; i++) {
      if (alg == getAlgorithmAcronym(Algorithm(i))) {
        return Algorithm(i);
      }
    }
    return kAlgorithmUnknown;
  }

  static std::vector<std::string> getKernelNamesOrderedVector() {
    return {"naiveSK",     "naiveMK",     "regenerationSK",
            "streamingMK", "streamingSK", "sortingSK"};
  }

  static auto getKernelAcronym(Kernel ker) -> const std::string {
    auto kernels = getKernelNamesOrderedVector();
    assert(kernels.size() == Kernel::kKernelUnknown &&
           "check the ConfigParser class Kernel, kUnknown must be at the end");
    if (ker > kernels.size()) {
      LOG_CONFIG(" kernel not present in the list of kernels");
    }
    if (ker < 0) {
      LOG_CONFIG(" error value < 0");
    }
    return kernels[ker];
  }

  static auto getKernel(std::string ker) -> Kernel {
    for (int i = 0; i < Kernel::kKernelUnknown; i++) {
      if (ker == getKernelAcronym(Kernel(i))) {
        return Kernel(i);
      }
    }
    return kKernelUnknown;
  }

  friend auto operator<<(std::ostream& str, Config const& c) -> std::ostream& {
    str << "algorithm_" << c.getAlgorithmAcronym(c.algorithm) << "_kernel_"
        << c.getKernelAcronym(c.kernel) << "_iter_"
        << c.path_tracing_config.iterations;
    return str;
  }

  [[nodiscard]] auto toString() const -> std::string {
    std::ostringstream sstr;
    sstr << *this;
    return sstr.str();
  }
};

#endif /* CONFIG_H_ */
