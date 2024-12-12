#pragma once

#include "Config.h"
#include "Defines.h"
#include <cuda_runtime.h>

double calculateOccupancy(CudaConfig cuda_config_) {

	float warp_size = cuda_config_.device_properties.warpSize;
	int block_size = cuda_config_.block_size.x *  cuda_config_.block_size.y *  cuda_config_.block_size.z;
	int n_blocks = cuda_config_.grid_size.x * cuda_config_.grid_size.y * cuda_config_.grid_size.z ;

	float active_warps = n_blocks * block_size / warp_size;
	float maxWarps = cuda_config_.device_properties.multiProcessorCount  * cuda_config_.device_properties.maxThreadsPerMultiProcessor / warp_size;

	return (double)active_warps / maxWarps * 100;
}

void maxOccupancyGrid(CudaConfig& cuda_config_, void* kernel) {

	//grid size
	int n_blocks;

#ifdef MAXIMAZE_OCCUPANCY
	n_blocks = MIN_BLOCKS_PER_MULTIPROCESSOR;
#else
	int block_size = cuda_config_.block_size.x *  cuda_config_.block_size.y *  cuda_config_.block_size.z;

	cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		&n_blocks,
		kernel,
		block_size,
		cuda_config_.dynamic_shared_memory
	);
#endif

	cuda_config_.grid_size.x = n_blocks * cuda_config_.device_properties.multiProcessorCount;
}

template <typename UnaryFunctor>
void maxOccupancyConfig(CudaConfig& cuda_config_, void* kernel, UnaryFunctor shared_mem_functor, int max_block_size = 0) {

	int min_grid_size;
	int block_size;

	//block size
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(
		&min_grid_size,
		&block_size,
		kernel,
		shared_mem_functor,
		max_block_size
	);

	cuda_config_.dynamic_shared_memory = shared_mem_functor(block_size);
	cuda_config_.block_size = dim3(block_size, 1, 1);

#ifdef MAXIMAZE_OCCUPANCY
	cuda_config_.grid_size = MIN_BLOCKS_PER_MULTIPROCESSOR * cuda_config_.device_properties.multiProcessorCount;
#else
	cuda_config_.grid_size = dim3(min_grid_size, 1, 1);
#endif

}

template<>
void maxOccupancyConfig<int>(CudaConfig& cuda_config_, void* kernel, int shared_mem, int max_block_size) {

	int min_grid_size;
	int block_size;

	//shared memory
	cuda_config_.dynamic_shared_memory = shared_mem;

	//block size
	cudaOccupancyMaxPotentialBlockSize(
		&min_grid_size,
		&block_size,
		kernel,
		cuda_config_.dynamic_shared_memory,
		max_block_size
	);

	cuda_config_.block_size = dim3(block_size);
	cuda_config_.grid_size = dim3(min_grid_size);
}
