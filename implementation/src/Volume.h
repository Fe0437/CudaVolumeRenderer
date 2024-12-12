#pragma once

#include <cuda_runtime.h>
#include "Math.h"
#include "Geometry.h"
#include "Rng.h"
#include "HG.h"

struct PhaseFunction {
	__host__ __device__
	virtual float3 sample(const float3& dir, Rng& rng) const = 0;
};

struct HG {
	float g = 0.f;

	__host__ __device__ HG() {}
	__host__ __device__ __forceinline__
		inline float3 sample(const float3& dir, Rng& rng) const {
			float2 rnd = rng.getFloat2();
			return ImportanceSampleHG(dir, g, rnd.x, rnd.y);
	}
};

template <class VolumeType>
struct DeviceVolume {

	uint3 grid_resolution;
	cudaTextureObject_t volume_tex;

	 __device__ __forceinline__ float3 volumeToGrid(float3 p) {
		 p.x = p.x * (grid_resolution.x - 1);
		 p.y = p.y * (grid_resolution.y - 1);
		 p.z = p.z * (grid_resolution.z - 1);
		return p;
	}

	__device__ __forceinline__ VolumeType operator()(float3 p) {

		float3 coord = volumeToGrid(p);

	#ifdef MITSUBA_COMPARABLE
			const int x1 = floorf(coord.x),
				y1 = floorf(coord.y),
				z1 = floorf(coord.z),
				x2 = x1 + 1, y2 = y1 + 1, z2 = z1 + 1;

			const float fx = coord.x - x1, fy = coord.y - y1, fz = coord.z - z1,
				_fx = 1.0f - fx, _fy = 1.0f - fy, _fz = 1.0f - fz;

			const VolumeType
				d000 = get(x1, y1, z1),
				d001 = get(x2, y1, z1),
				d010 = get(x1, y2, z1),
				d011 = get(x2, y2, z1),
				d100 = get(x1, y1, z2),
				d101 = get(x2, y1, z2),
				d110 = get(x1, y2, z2),
				d111 = get(x2, y2, z2);

			return ((d000*_fx + d001*fx)*_fy +
				(d010*_fx + d011*fx)*fy)*_fz +
				((d100*_fx + d101*fx)*_fy +
				(d110*_fx + d111*fx)*fy)*fz;
	#else
			return get(int(coord.x), int(coord.y), int(coord.z));
	#endif

}

	__device__ __forceinline__ VolumeType get(uint x, uint y, uint z);
	__device__ __forceinline__ VolumeType get(uint x);
};


template <class VolumeType>
struct HostDeviceVolume {

	uint3 grid_resolution;
	cudaTextureObject_t volume_tex;

	__device__ __forceinline__ float3 volumeToGrid(float3 p) {
		p.x = p.x * (grid_resolution.x - 1);
		p.y = p.y * (grid_resolution.y - 1);
		p.z = p.z * (grid_resolution.z - 1);
		return p;
	}

	__device__ __forceinline__ VolumeType operator()(float3 p) {

		float3 coord = volumeToGrid(p);

#ifdef MITSUBA_COMPARABLE

		const int x1 = floorf(coord.x),
			y1 = floorf(coord.y),
			z1 = floorf(coord.z),
			x2 = x1 + 1, y2 = y1 + 1, z2 = z1 + 1;

		const float fx = coord.x - x1, fy = coord.y - y1, fz = coord.z - z1,
			_fx = 1.0f - fx, _fy = 1.0f - fy, _fz = 1.0f - fz;

		const VolumeType
			d000 = get(x1, y1, z1),
			d001 = get(x2, y1, z1),
			d010 = get(x1, y2, z1),
			d011 = get(x2, y2, z1),
			d100 = get(x1, y1, z2),
			d101 = get(x2, y1, z2),
			d110 = get(x1, y2, z2),
			d111 = get(x2, y2, z2);

		return ((d000*_fx + d001*fx)*_fy +
			(d010*_fx + d011*fx)*fy)*_fz +
			((d100*_fx + d101*fx)*_fy +
			(d110*_fx + d111*fx)*fy)*fz;
#else
		return get(int(coord.x), int(coord.y), int(coord.z));
#endif

	}

	__device__ __forceinline__ VolumeType get(uint x, uint y, uint z);
	__device__ __forceinline__ VolumeType get(uint x);
};


template <class VOLUME_TYPE>
class Volume {
public:
	typedef VOLUME_TYPE VolumeType;

private:
	VolumeType* vol_data_;

public:
	uint3 grid_resolution;

	Volume(
		VolumeType* vol_data,
		uint3 _grid_resolution
	) :
		vol_data_(vol_data),
		grid_resolution(_grid_resolution)
	{}

	Volume() :vol_data_(0) {}

	VolumeType* getVolumeData() { return vol_data_; }
	uint3 getVolumeSize() { return grid_resolution; }
	
	VolumeType operator()(float3 p) {
		return vol_data_[int(p.x) + grid_resolution.x*int(p.y) + grid_resolution.x*grid_resolution.y*int(p.z)];
	}

	size_t getBytes() {
		return grid_resolution.x * grid_resolution.y * grid_resolution.z * sizeof(VolumeType);
	}

	cudaExtent getCudaExtent() {
		cudaExtent extent;
		extent.width = grid_resolution.x;
		extent.height = grid_resolution.y;
		extent.depth = grid_resolution.z;
		return extent;
	}

	void ZYXToMortonOrder() {
		VolumeType* temp = (VolumeType*) malloc(grid_resolution.x * grid_resolution.y * grid_resolution.z * sizeof(VolumeType));
		auto* vol_data_iter = vol_data_;
		for (int i = 0; i < grid_resolution.x; i++) {
			for (int j = 0; j < grid_resolution.y; j++) {
				for (int k = 0; k < grid_resolution.z; k++) {
					int id = getMortonIndex(i, grid_resolution.x, j, grid_resolution.y, k, grid_resolution.z);
					temp[id] = *vol_data_iter;
					vol_data_iter++;
				}
			}
		}
		free(vol_data_);
		vol_data_ = temp;
	}
};




