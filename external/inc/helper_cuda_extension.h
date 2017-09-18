/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions for initialization and error checking

#ifndef HELPER_CUDA_EXT_H
#define HELPER_CUDA_EXT_H

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <helper_cuda.h>

#ifdef __CUDA_RUNTIME_H__

// Initialization code to find the best CUDA Device
	inline int findCudaDevice(int devID=-1)
	{
		cudaDeviceProp deviceProp;
		if (devID < 0)
		{
			//pick the device with highest Gflops/s
			devID = gpuGetMaxGflopsDeviceId();
			checkCudaErrors(cudaSetDevice(devID));
			checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
			printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
		}
		else
		{
			devID = gpuDeviceInit(devID);
			if (devID < 0)
			{
				printf("gpuDeviceInit error - exiting...\n");
				exit(EXIT_FAILURE);
			}
		}
		return devID;
	}

#endif

// end of CUDA Helper Functions


#endif
