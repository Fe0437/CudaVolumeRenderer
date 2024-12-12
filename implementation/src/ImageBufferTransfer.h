#ifndef IMAGE_BUFFER_TRANSFER_H_
#define IMAGE_BUFFER_TRANSFER_H_

#pragma once

#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "Math.h"

#include "AbstractRenderer.h"

namespace UtilityFunctors{
struct Scale; 
}

class HostImageBufferTansferDelegate : public Buffer2DTransferDelegate<UtilityFunctors::Scale> {

public:
	virtual void transfer(void* buffer_in, size_t pitch_in, void* buffer_out, size_t offset_out, size_t pitch_out, size_t width, size_t height, UtilityFunctors::Scale transform);
};

class DeviceImageBufferTansferDelegate : public Buffer2DTransferDelegate<UtilityFunctors::Scale> {

public:

	virtual void transfer(void* buffer_in, size_t pitch_in, void* buffer_out, size_t offset_out, size_t pitch_out, size_t width, size_t height, UtilityFunctors::Scale transform);
};


class DeviceTiledImageBufferTansferDelegate : public Buffer2DTransferDelegate<UtilityFunctors::Scale> {
	
	float* d_buffer_transfer_ = 0;

public:

	DeviceTiledImageBufferTansferDelegate(size_t size) {
		checkCudaErrors(cudaMalloc(&d_buffer_transfer_, size));
	}

	virtual void transfer(void* buffer_in, size_t pitch_in, void* buffer_out, size_t offset_out, size_t pitch_out, size_t width, size_t height, UtilityFunctors::Scale transform);
};

#endif // !IMAGE_BUFFER_TRANSFER_H_
