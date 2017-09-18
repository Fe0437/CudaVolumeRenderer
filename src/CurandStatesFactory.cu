/*
 * CurandStatesFactory.cu
 *
 *  Created on: 03/set/2017
 *      Author: macbook
 */

#include "CurandStatesFactory.cuh"


CurandStatesFactory::CurandStatesFactory(dim3 blocks, dim3 threads_per_block):
	blocks_(blocks),
	threads_per_block_(threads_per_block),
	width_(1),
	height_(1),
	samples_(1)
{
}

/** \brief
 * function that create the curandStateXORWOW inside the device and return the pointer to it
 * the function sets also the width height and number of samples which defines the total number of random sequence
 * which are necessary.
 */
curandStateXORWOW_t* CurandStatesFactory::createCurandStatesOnDevice(unsigned int _width,unsigned int _height,unsigned int _samples){
	width_ = _width;
	height_ = _height;
	samples_ = _samples;

	curandStateXORWOW_t *d_states;
	unsigned int* seeds = new unsigned int[width_ * height_ * samples_];

	unsigned int *d_seeds;
	checkCudaErrors(cudaMalloc(&d_seeds, sizeof(unsigned int) * width_ * height_ * samples_));
	cudaMalloc(&d_states, sizeof(curandStateXORWOW_t) * width_ * height_ * samples_);

	initRandomStates(seeds, d_seeds, d_states);
	cudaFree(d_seeds);
	delete[] seeds;
	return d_states;
}

namespace XORShift { // XOR shift PRNG
	unsigned int x = 123456789;
	unsigned int y = 362436069;
	unsigned int z = 521288629;
	unsigned int w = 88675123;
	inline unsigned int frand() {
	unsigned int t;
			t = x ^ (x << 11);
			x = y; y = z; z = w;
	return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
		}
}

void CurandStatesFactory::initSeedsWithXORShift(unsigned int *seeds){
	for (int i = 0; i < width_ * height_* samples_; ++i)
			seeds[i] = XORShift::frand();
}

void CurandStatesFactory::initDeviceSeedsWithSeeds(unsigned int *seeds, unsigned int *d_seeds){
	cudaMemcpy(d_seeds, seeds, sizeof(unsigned int) * width_ * height_ * samples_, cudaMemcpyHostToDevice);
}

void CurandStatesFactory::initRandomStates(unsigned int *seeds, unsigned int *d_seeds, curandStateXORWOW_t *d_states){
				initSeedsWithXORShift(seeds);
				initDeviceSeedsWithSeeds(seeds,d_seeds);
				initCurandStates(d_seeds, d_states);
			}

//CurandStates init cuda kernel
__global__ void initCurandStatesKernel(unsigned int *d_seeds, curandStateXORWOW_t *d_states, unsigned int width, unsigned int height, unsigned int samples) {
		int ix = threadIdx.x + blockIdx.x * blockDim.x;
		int iy = threadIdx.y + blockIdx.y * blockDim.y;
		if (ix >= width || iy >= height)
			return;
		int idx = ix + iy * width;
		for(int i=0; i<samples; i++)
			curand_init(d_seeds[idx*samples + i], idx*0, 0, &d_states[idx*samples + i]);
	}

void CurandStatesFactory::initCurandStates(unsigned int *d_seeds, curandStateXORWOW_t *d_states) {
		initCurandStatesKernel<<<blocks_, threads_per_block_>>>(d_seeds, d_states, width_, height_, samples_);
	}


