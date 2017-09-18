/*
 * Random.h
 *
 *  Created on: 01/set/2017
 *      Author: macbook
 */

#ifndef CURANDSTATESFACTORY_H_
#define CURANDSTATESFACTORY_H_

#ifdef WIN32
#include <windows.h>
#endif

#include "helper_cuda.h"

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <curand_kernel.h>

class CurandStatesFactory{

	dim3 blocks_, threads_per_block_;

	unsigned int width_, height_, samples_;

public:

	CurandStatesFactory(dim3 blocks, dim3 threads_per_block);

	/** \brief
	 * function that create the curandStateXORWOW inside the device and return the pointer to it
	 * the function sets also the width height and number of samples which defines the total number of random sequence
	 * which are necessary.
	 */
	curandStateXORWOW_t* createCurandStatesOnDevice(unsigned int _width,unsigned int _height=1,unsigned int _samples=1);

private:

	void initSeedsWithXORShift(unsigned int *seeds);
	void initDeviceSeedsWithSeeds(unsigned int *seeds, unsigned int *d_seeds);
	void initRandomStates(unsigned int *seeds, unsigned int *d_seeds, curandStateXORWOW_t *d_states);
	void initCurandStates(unsigned int *d_seeds, curandStateXORWOW_t *d_states);
};


#endif /* CURANDSTATESFACTORY_H_ */
