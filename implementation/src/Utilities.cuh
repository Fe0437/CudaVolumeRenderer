#pragma once

// 1. CUDA headers first
#include <cuda_runtime.h>

// 2. Third-party libraries
#include <cub/cub.cuh>

// 4. Project headers
#include "CVRMath.h"
#include "Ray.h"
#include "Rng.h"
#include "Utilities.h"

__device__ inline float4 atomicVectorAdd(float4* current_value, float4 value) {
  float4 ret;
  ret.x = atomicAdd(&(current_value->x), value.x);
  ret.y = atomicAdd(&(current_value->y), value.y);
  ret.z = atomicAdd(&(current_value->z), value.z);
  ret.w = current_value->w = 1.f;
  return ret;
}

// used instead of return atomicAdd(ctr, 1);
// from
// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
#if CUDART_VERSION >= 9000

#include <cooperative_groups.h>

// warp-aggregated atomic increment

template <typename T>
__device__ __forceinline__ int atomicAggInc(T* counter) {
  using namespace cooperative_groups;

  coalesced_group active = coalesced_threads();

  int mask = active.ballot(1);
  // select the leader
  int leader = __ffs(mask) - 1;

  // leader does the update
  int res = 0;
  if (active.thread_rank() == leader) {
    res = atomicAdd(counter, __popc(mask));
  }

  // broadcast result
  res = active.shfl(res, leader);

  // each thread computes its own value
  return res + __popc(mask & ((1 << active.thread_rank()) - 1));
}

/*__device__ __inline__ int atomicAggInc(unsigned int *ctr) {

        auto g = coalesced_threads();
        unsigned int warp_res;
        if (g.thread_rank() == 0)
                warp_res = atomicAdd(ctr, g.size());
        return g.shfl(warp_res, 0) + g.thread_rank();
}*/

#else

template <typename T>
__device__ int atomicAggInc(T* ctr) {
  return atomicAdd(ctr, 1);
}
#endif

namespace CubUtilities {

template <int ITEMS_PER_THREAD>
__device__ __forceinline__ void IndexOfLoadDirectWarpStriped(
    int linear_tid,  ///< [in] A suitable 1D thread-identifier for the calling
                     ///< thread (e.g., <tt>(threadIdx.y * blockDim.x) +
                     ///< linear_tid</tt> for 2D thread blocks)
    int (&index)[ITEMS_PER_THREAD]) {
  int tid = linear_tid & (CUB_PTX_WARP_THREADS - 1);
  int wid = linear_tid >> CUB_PTX_LOG_WARP_THREADS;
  int warp_offset = wid * CUB_PTX_WARP_THREADS * ITEMS_PER_THREAD;

  int start_index = warp_offset + tid;

  //	warp-striped order
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    index[ITEM] = start_index + (ITEM * CUB_PTX_WARP_THREADS);
  }
}

template <int BLOCK_THREADS,
          int ITEMS_PER_THREAD>
__device__ __forceinline__ void IndexOfLoadDirectStriped(
    int linear_tid,  ///< [in] A suitable 1D thread-identifier for the calling
                     ///< thread (e.g., <tt>(threadIdx.y * blockDim.x) +
                     ///< linear_tid</tt> for 2D thread blocks)
    int (&index)[ITEMS_PER_THREAD]) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    index[ITEM] = linear_tid + (ITEM * BLOCK_THREADS);
  }
}

template <int ITEMS_PER_THREAD>
__device__ __forceinline__ void IndexOfLoadDirect(
    int linear_tid,  ///< [in] A suitable 1D thread-identifier for the calling
                     ///< thread (e.g., <tt>(threadIdx.y * blockDim.x) +
                     ///< linear_tid</tt> for 2D thread blocks)
    int (&index)[ITEMS_PER_THREAD]) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    index[ITEM] = ITEM + (linear_tid * ITEMS_PER_THREAD);
  }
}

};  // namespace CubUtilities

//------------------------------------------------------distance
//sampling--------------------------------------------------------------------

// analytical
__device__ __forceinline__ inline float sampleDistance(float e, float sigma) {
  return -logf(1.0f - e) / sigma;
}

__device__ __forceinline__ float3 worldToAABB(float3 p, float3 range,
                                              float3 start) {
  return (p - start / range);
}

__device__ __forceinline__ float woodcockStep(float _dMaxInv, float _xi) {
  return -logf(max(_xi, EPSILON)) * _dMaxInv;
}

template <typename DensityVolume>
__device__ __forceinline__ float woodcockTracking(
    float3 origin, float3 dir, float max_t, float max_density,
    DensityVolume density_volume, float3 extent, float3 start,
    float density_scale, Rng* rng) {
  float inv_max_sigmat = 1.0f / (density_scale * max_density);
  float event_density = 0.0f;
  float t = 0.0;
  float3 coord;
  do {
    t += woodcockStep(inv_max_sigmat, rng->getFloat());
    coord = worldToAABB(origin + (t * dir), extent, start);
    event_density = density_scale * density_volume(coord);

  } while (t <= max_t && event_density * inv_max_sigmat < rng->getFloat());

  return t;
}

__host__ __device__ inline unsigned int utilhash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);

  a = (a ^ 0xc761c23c) ^ (a >> 19);

  a = (a + 0x165667b1) + (a << 5);

  a = (a + 0xd3a2646c) ^ (a << 9);

  a = (a + 0xfd7046c5) + (a << 3);

  a = (a ^ 0xb55a4f09) ^ (a >> 16);

  return a;
}

__device__ inline Rng makeSeededRng(int current_iteration, int index,
                                    int depth) {
  int h =
      utilhash((1 << 31) | (depth << 22) | current_iteration) ^ utilhash(index);
  return Rng(h);
}

inline __device__ float2 indexToRaster(const float2 resolution,
                                       float2 pixel_coord, Rng& rng) {
  pixel_coord = pixel_coord + rng.getFloat2();
  return (pixel_coord * 2.f / resolution) - 1.0f;
}

inline __device__ Ray cameraGenerateRay(float2& raster, float2& raster_to_view,
                                        float3x4& inv_view_mat) {
  // calculate eye ray in world space
  Ray eyeRay;
  raster = raster_to_view * raster;
  eyeRay.o =
      make_float3(mul(inv_view_mat, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
  eyeRay.d = normalize(make_float3(raster, 1.0f));
  eyeRay.d = mul(inv_view_mat, eyeRay.d);

  // orthographic camera (not working for mitsbua)
  /*eyeRay.o = make_float3(mul(c_inv_view_mat, make_float4(raster.x, raster.y,
  0.0f, 1.0f))); eyeRay.d = normalize(make_float3(0, 0, 1.0f)); eyeRay.d =
  mul(c_inv_view_mat, eyeRay.d);*/

  return eyeRay;
}

inline __host__ int iDivUp(int a, int b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

inline __device__ Ray indexToCameraRay(float2 pixel_coord, float2& resolution,
                                       float2& raster_to_view,
                                       float3x4& inv_view_mat, Rng& rng) {
  float2 raster = indexToRaster(resolution, pixel_coord, rng);
  return cameraGenerateRay(raster, raster_to_view, inv_view_mat);
}
