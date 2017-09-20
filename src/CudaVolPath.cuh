/*
 * CudaVolPath.h
 *
 *  Created on: 30/ago/2017
 *      Author: macbook
 */

#ifndef CUDAVOLPATH_H_
#define CUDAVOLPATH_H_

#ifdef WIN32
#include <windows.h>
#endif

#include "Integrator.h"

#include "helper_cuda.h"
#include "helper_math.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>
#include "UtilMathStructs.h"

#include <thrust/remove.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Config.h"
#include "RenderCamera.h"
#include "Scene.h"
#include "CurandStatesFactory.cuh"
#include "Defines.h"
#include "Image.h"

using namespace utilityCore;
using namespace std;

struct Ray
	{
	    float3 o;   // origin
	    float3 d;   // direction
	    float4 throughput;
	};

__global__ void
d_traceRay(
		Ray* d_rays,
		int* d_activePixels,
		int nActivePixels,
		float4 *d_output,
		float3 box_min,
		float3 box_max,
		float brightness,
		float max_density,
		curandStateXORWOW_t *d_states,
		int steps_per_call,
		int samples,
		cudaTextureObject_t albedo_tex,
		cudaTextureObject_t density_tex
		);

__global__ void d_initRays(
		Ray* d_rays,
		int* d_active_pixels,
		float4* d_output,
		float2 tile_start,
		float2 tile_dim,
		float2 resolution,
		float3 box_min,
		float3 box_max,
		curandStateXORWOW_t *d_states,
		int sample_per_pass,
		int samples
		);

class CudaVolPath: public Integrator {

	const Config& config_;
	RenderCamera camera_;

	uint batch_bounces_;
	glm::ivec2 rendering_resolution_;
	glm::ivec2 n_tiles_;
	glm::ivec2 tile_dim_;
	uint n_active_pixels_;

	//device variables to set
	int* d_active_pixels_ = 0;
	Ray* d_rays_ = 0;
	float4* d_output_ = 0;
	cudaArray *d_albedo_array_ = 0;
	cudaArray *d_density_array_ = 0;
	curandStateXORWOW_t *d_states_ = 0;

	cudaTextureObject_t albedo_tex_;
	cudaTextureObject_t density_tex_;

public:
	CudaVolPath(const Config& config, const glm::ivec2& n_tiles);
	virtual ~CudaVolPath();

	void copyInvViewMatrix(float* invViewMatrix, size_t sizeofMatrix);

	void render();
	void saveImage();

private:

	template < class VolumeType>
		cudaTextureObject_t createTextureWithVolume(cudaArray * d_array, VolumeType *h_volume, cudaExtent volumeSize);
	void initCamera();
	void initRandom();
	void initRendering();
	void renderingLoop();
	void initTextures();
	void releaseDeviceMemory();

};

#endif /* CUDAVOLPATH_H_ */
