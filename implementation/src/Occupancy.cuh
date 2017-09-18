#pragma once

// 1. CUDA headers first
#include <cuda_runtime.h>

#include "Config.h"
#include "Defines.h"

inline double calculateOccupancy(CudaConfig cuda_config_) {
  float warp_size = cuda_config_.device_properties.warpSize;
  int block_size = cuda_config_.block_size.x * cuda_config_.block_size.y *
                   cuda_config_.block_size.z;
  int n_blocks = cuda_config_.grid_size.x * cuda_config_.grid_size.y *
                 cuda_config_.grid_size.z;

  float active_warps = n_blocks * block_size / warp_size;
  float maxWarps = cuda_config_.device_properties.multiProcessorCount *
                   cuda_config_.device_properties.maxThreadsPerMultiProcessor /
                   warp_size;

  return (double)active_warps / maxWarps * 100;
}

inline void maxOccupancyGrid(CudaConfig& cuda_config_, void* kernel) {
  // grid size
  int n_blocks;

#ifdef MAXIMAZE_OCCUPANCY
  n_blocks = MIN_BLOCKS_PER_MULTIPROCESSOR;
#else
  int block_size = cuda_config_.block_size.x * cuda_config_.block_size.y *
                   cuda_config_.block_size.z;

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &n_blocks, kernel, block_size, cuda_config_.dynamic_shared_memory);
#endif

  cuda_config_.grid_size.x =
      n_blocks * cuda_config_.device_properties.multiProcessorCount;
}

template <typename F>
inline void maxOccupancyConfig(CudaConfig& cuda_config_, F kernel,
                               size_t dynamic_shared_memory,
                               int block_size_limit = 0) {
  int min_grid_size;
  int block_size;

  // Get optimal block size and minimum grid size
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel,
                                     dynamic_shared_memory, block_size_limit);

  // Get max blocks per SM for this configuration
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &cuda_config_.max_active_blocks_per_sm, kernel, block_size,
      dynamic_shared_memory);

  // Get device properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, cuda_config_.devId);

  // Calculate total max blocks across all SMs
  cuda_config_.max_total_blocks =
      cuda_config_.max_active_blocks_per_sm * prop.multiProcessorCount;

  // Set the configuration
  cuda_config_.block_size = dim3(block_size);
  cuda_config_.grid_size = dim3(min_grid_size);
  cuda_config_.dynamic_shared_memory = dynamic_shared_memory;
}