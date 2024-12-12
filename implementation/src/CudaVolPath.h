/*
 * CudaVolPath.h
 *
 *  Created on: 30/ago/2017
 *      Author: macbook
 */
#pragma once

#ifndef CUDAVOLPATH_H_
#define CUDAVOLPATH_H_

#include "AbstractRenderer.h"

#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "helper_math.h"

#include "Math.h"
#include "Config.h"
#include "Defines.h"
#include "Rng.h"
#include "Ray.h"
#include "Utilities.h"
#include "Debug.h"

#include "RenderKernelLauncher.h"

#include <boost/iterator/counting_iterator.hpp>


namespace UtilityFunctors {
	struct Scale;
}

template <class VolPathKernelLauncher>
class CudaVolPath: public AbstractProgressiveRenderer{

public: 
	typedef Buffer2DTransferDelegate<UtilityFunctors::Scale> OutputDelegate;

	struct VolumeDataPointer {
		cudaArray* cuda_array = 0;
		void* cuda_malloc = 0;
	};

	typedef std::vector<uint2> TileArray;

private:
	//configurations
	PathTracingConfig path_tracing_config_;
	TilingConfig tiling_config_;
	CudaConfig cuda_config_;

	Scene scene_;
	uint current_iteration_;

	TileArray::iterator current_tile_;
	TileArray tiles_;

	cudaEvent_t current_buffer_ready_event_, processing_buffer_ready_event_;

	float4* d_output_ = 0;
	float4* d_output_processed_ = 0;

	VolumeDataPointer d_albedo_;
	VolumeDataPointer d_density_;

	OutputDelegate* output_delegate_ = 0;
	VolPathKernelLauncher kernel_launcher_;

	std::vector<cudaTextureObject_t> volume_textures_;

public:
	CudaVolPath(const Config& config, OutputDelegate* output_delegate);
    virtual ~CudaVolPath();

	void setOutputBufferTransferDelegate(OutputDelegate* bdelegate) {
		delete output_delegate_;
		output_delegate_ = bdelegate;
	}

	//progressive rendering interface
	void initRendering();
	void runIterations();
	void setNIterations(uint iterations);
	bool imageComplete();
	void getImage(void* buffer_out);

	//rendering interface
	void render(void* buffer_out);

private:

	template < class VolumeType>
		cudaTextureObject_t createTextureWithVolume(VolumeDataPointer d_array, VolumeType *h_volume, cudaExtent volumeSize);
	void initCamera();
	void initRenderState();
	void prepareForNextIterations();
	void allocateDeviceMemory();
	void initDeviceScene();
	void releaseDeviceMemory();
	void initTileArray();
};

#endif /* CUDAVOLPATH_H_ */
