#ifndef RENDER_KERNEL_LAUNCHER_H_
#define RENDER_KERNEL_LAUNCHER_H_

#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cuda_profiler_api.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include "Medium.h"
#include "Ray.h"
#include "Config.h"
#include "Debug.h"
#include "Bsdf.h"

class RenderKernelLauncher {

protected:
	typedef unsigned int uint;

	CudaConfig cuda_config_;
	uint2 resolution_{ 0,0 };
	float4* d_output_ = 0;

public:
	RenderKernelLauncher(){}

	void setOutputPtr(float4* d_output) { 
		d_output_ = d_output;
	}

	void setResolution(uint2 resolution) { 
		resolution_ = resolution;
		copyResolution(make_float2(resolution.x, resolution.y));
	}

	void setCudaConfig(CudaConfig cuda_config) { cuda_config_ = cuda_config; }

	virtual void allocateDeviceMemory() {};
	virtual void init() {}
	virtual void launchRender() = 0;
	virtual void reset() {
		cudaDeviceSynchronize();
	}
	virtual void releaseDeviceMemory(){};
	void loadStatistics();
	void saveStatistics();
	void copyInvViewMatrix(float* inv_view_mat, size_t size_of_mat);
	void copyRasterToView(float2 raster_to_view);
	void copyResolution(float2 resolution);
	void copyPixelIndexRange(float2 pixel_index_range);
	void copyOffset(uint2 offset);
};

template <class DEVICE_SCENE>
class VolPTKernelLauncher : public RenderKernelLauncher {
public:
	typedef DEVICE_SCENE DeviceScene;
protected:
	DeviceScene device_scene_;
	uint n_iterations_ = 1;
	uint n_paths_ = 0;

public:
	VolPTKernelLauncher(){}
	virtual void launchRender() = 0;
	virtual void setNIterations(uint n_iterations);
	uint getNIterations() { return n_iterations_; }	
	void setScene(const DeviceScene& device_scene) { device_scene_ = device_scene; }
	DeviceScene& getScene() { return device_scene_; }
};


template <class DEVICE_SCENE>
class NaiveVolPTsk : public VolPTKernelLauncher<DEVICE_SCENE> {

public:
	NaiveVolPTsk() {}

	virtual void init();
	virtual void launchRender();

};


template <class DEVICE_SCENE>
class NaiveVolPTmk : public VolPTKernelLauncher<DEVICE_SCENE>{

	uint current_iteration_ = 0;
	int* d_active_pixels_ = 0;
	uint n_processed_pixels_ = 0;
	uint n_active_pixels_ = 0;
	Path* d_paths_ = 0;

public:
	NaiveVolPTmk() {}

	virtual void init();

	virtual void allocateDeviceMemory();

	virtual void launchRender();

	virtual void releaseDeviceMemory();

private:
	void extend(uint path_length, uint current_iteration);
};


template <class DEVICE_SCENE>
class RegenerationVolPTsk : public VolPTKernelLauncher<DEVICE_SCENE> {
	
	uint seed_ = 0;

public:
	RegenerationVolPTsk() {}

	virtual void init();
	virtual void launchRender();
	virtual void reset();
};


template <class DEVICE_SCENE>
class StreamingVolPTmk : public VolPTKernelLauncher<DEVICE_SCENE> {
	
	uint seed_ = 0;
	uint n_threads_ = 0;
	
	// arrays for the simple method
	//Thread* d_threads_in_ = 0;
	//Thread* d_threads_out_ = 0;

	Rng::State* states_;
	bool* active_threads_;

	//SOA method
	Threads d_threads_in_{ 0 };
	Threads d_threads_out_{ 0 };

public:
	StreamingVolPTmk() {}

	virtual void allocateDeviceMemory();
	virtual void init();
	virtual void launchRender();
	virtual void reset();
	virtual void releaseDeviceMemory();

};

template <class DEVICE_SCENE>
class StreamingVolPTsk : public VolPTKernelLauncher<DEVICE_SCENE> {
	uint seed_ = 0;
	uint n_threads_ = 0;
	Threads d_threads_{ 0 };

public:
	StreamingVolPTsk() {}

	virtual void allocateDeviceMemory();
	virtual void init();
	virtual void launchRender();
	virtual void reset();
	virtual void releaseDeviceMemory();

};


template <class DEVICE_SCENE>
class SortingVolPTsk : public VolPTKernelLauncher<DEVICE_SCENE> {
	uint seed_ = 0;
	uint n_threads_ = 0;
	Threads d_threads_{ 0 };

public:
	SortingVolPTsk() {}
	virtual void allocateDeviceMemory();
	virtual void init();
	virtual void launchRender();
	virtual void reset();
	virtual void releaseDeviceMemory();

};
#endif