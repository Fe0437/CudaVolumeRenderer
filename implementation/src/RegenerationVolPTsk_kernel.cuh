#ifndef REGENERATION_VOLPT_SK_KERNEL_H_
#define REGENERATION_VOLPT_SK_KERNEL_H_
#pragma once

#include <cuda_runtime.h>

#include "Bsdf.h"
#include "CVRMath.h"
#include "Geometry.h"
#include "Medium.h"
#include "Ray.h"
#include "Rng.h"
#include "Utilities.cuh"
#include "helper_cuda.h"

namespace RegenerationVolPTsk_kernel {

__device__ uint seed = 0;
__device__ uint paths_head_global = 0;

template <typename Scene, typename Isect = typename Scene::SceneIsect>
KERNEL_LAUNCH d_render(float4* d_output, Scene scene) {
#if CUDART_VERSION < 9000
  // dynamically allocated with blockDim.y
  volatile extern __shared__ uint paths_head_block[];
  volatile uint& head_warp = paths_head_block[threadIdx.y];
  int tid = threadIdx.x + (blockDim.x * threadIdx.y) +
            (blockDim.x * blockDim.y) * blockIdx.x;

#else
  int tid = threadIdx.x + (blockDim.x * blockIdx.x);
  int laneId = threadIdx.x & 0x1f;
  int head_warp;
#endif

  // persistent thread id
  bool active = false;

  uint image_id;
  Path path;
  Isect isect;
  Rng rng(seed + tid);

  // persistent thread loop
  do {
    // check if all the threads in a warp are idle
    if (!any(active)) {
      //---------------REGENERATE--------------------

      // let lane 0 fetch [wh, wh + WARP_SIZE - 1] for a warp
#if CUDART_VERSION < 9000
      if (threadIdx.x == 0) {
        head_warp = atomicAdd(&paths_head_global, blockDim.x);
      }
      // path index per thread in a warp
      uint path_id = head_warp + threadIdx.x;
#else
      if (laneId == 0)
        head_warp =
            atomicAdd(&paths_head_global, 32);  // all threads except lane 0

      head_warp = __shfl_sync(
          0xffffffff, head_warp,
          0);  // Synchronize all threads in warp, and get "value" from lane 0

      // path index per thread in a warp
      uint path_id = head_warp + laneId;
#endif

      if (path_id >= c_n_paths) {
        return;
      }

      image_id = (path_id % (uint)(c_resolution.x * c_resolution.y));

      float2 pixel_index;
      pixel_index.x = (float)(image_id % ((uint)c_resolution.x)) + c_offset.x;
      pixel_index.y = (floorf((float)image_id / c_resolution.x)) + c_offset.y;

      path.ray = indexToCameraRay(pixel_index, c_pixel_index_range,
                                  c_raster_to_view, c_inv_view_mat, rng);
      path.throughput = make_float4(1.f, 1.f, 1.f, 1.f);

      active = true;
    }

    // the thread is active
    if (active) {
#ifdef RAYS_STATISTICS
      atomicAdd(&d_n_rays_statistics, 1);
#endif

      if (!scene.intersect(path.ray.o, path.ray.d, isect)) {
        atomicVectorAdd(
            &d_output[image_id],
            path.throughput * scene.Le(path.ray.o, path.ray.d, isect));
        active = false;
      } else {
        // intersecting volume BB
        float sampled_distance;
        auto* medium = scene.getMedium(isect);

        if (medium == 0 ||
            !medium->sampleDistance(path.ray.o, path.ray.d, isect.dist, rng,
                                    sampled_distance)) {
          // outside volume
          Frame frame;
          frame.setFromZ(isect.normal);
          float3 dir = frame.toLocal(normalize(-path.ray.d));

          path.ray.o = path.ray.o + path.ray.d * isect.dist;
          float weight = 1;

          if (scene.getBsdf(isect).sample(dir, path.ray.d, weight, rng)) {
            path.throughput *= weight;
            path.ray.d = frame.toWorld(path.ray.d);
            path.ray.o = path.ray.o + path.ray.d * EPSILON;
          }
        } else {
          path.ray.o =
              path.ray.o + path.ray.d * sampled_distance - path.ray.d * EPSILON;

          float4 albedo = medium->sampleAlbedo(path.ray.o);
          path.throughput = path.throughput * albedo;
          path.ray.d = medium->samplePhase(path.ray.d, rng);
        }
      }  // intersected medium

#ifdef RUSSIAN_ROULETTE
      /*
       *------roulette----
       */
      float pSurvive = fmin(1.f, fmaxf3(path.throughput));
      if (rng.getFloat() > pSurvive) {
        active = false;
      }
      path.throughput = path.throughput * 1.f / pSurvive;
#endif
    }
  } while (true);
}

///----------------------------------------------SINGLE THREAD REGENERATION
///KERNEL ---------------------------------------------------------

template <typename Scene, typename Isect = typename Scene::SceneIsect>
KERNEL_LAUNCH d_render_single_thread_regeneration(float4* d_output,
                                                  Scene scene) {
  // persistent thread id
  int tid = threadIdx.x + (blockDim.x * blockIdx.x);
  bool active = false;

  uint image_id;
  Path path;
  Isect isect;
  Rng rng(seed + tid);

  // persistent thread loop
  do {
    // check if all the threads in a warp are idle
    if (!active) {
      //---------------REGENERATE--------------------
      uint path_id = atomicAdd(&paths_head_global, 1);

      if (path_id >= c_n_paths) {
        return;
      }

      image_id = (path_id % (uint)(c_resolution.x * c_resolution.y));

      float2 pixel_index;
      pixel_index.x = (float)(image_id % ((uint)c_resolution.x)) + c_offset.x;
      pixel_index.y = (floorf((float)image_id / c_resolution.x)) + c_offset.y;

      path.ray = indexToCameraRay(pixel_index, c_pixel_index_range,
                                  c_raster_to_view, c_inv_view_mat, rng);
      path.throughput = make_float4(1.f, 1.f, 1.f, 1.f);

      active = true;
    }

#ifdef RAYS_STATISTICS
    atomicAdd(&d_n_rays_statistics, 1);
#endif
    if (!scene.intersect(path.ray.o, path.ray.d, isect)) {
      atomicVectorAdd(
          &d_output[image_id],
          path.throughput * scene.Le(path.ray.o, path.ray.d, isect));
      active = false;
    } else {
      // intersecting volume BB
      float sampled_distance;
      auto* medium = scene.getMedium(isect);

      if (medium == 0 ||
          !medium->sampleDistance(path.ray.o, path.ray.d, isect.dist, rng,
                                  sampled_distance)) {
        // outside volume
        Frame frame;
        frame.setFromZ(isect.normal);
        float3 dir = frame.toLocal(normalize(-path.ray.d));

        path.ray.o = path.ray.o + path.ray.d * isect.dist;
        float weight = 1;

        if (scene.getBsdf(isect).sample(dir, path.ray.d, weight, rng)) {
          path.throughput *= weight;
          path.ray.d = frame.toWorld(path.ray.d);
          path.ray.o = path.ray.o + path.ray.d * EPSILON;
        }
      } else {
        path.ray.o = path.ray.o + path.ray.d * sampled_distance;

        float4 albedo = medium->sampleAlbedo(path.ray.o);
        path.throughput = path.throughput * albedo;
        path.ray.d = medium->samplePhase(path.ray.d, rng);
      }
    }  // intersected medium

#ifdef RUSSIAN_ROULETTE
    /*
     *------roulette----
     */
    float pSurvive = fmin(1.f, fmaxf3(path.throughput));
    if (rng.getFloat() > pSurvive) {
      active = false;
    }
    path.throughput = path.throughput * 1.f / pSurvive;
#endif

  } while (true);
}

///----------------------------------------------BLOCK REGENERATION KERNEL
///---------------------------------------------------------

template <typename Scene, typename Isect = typename Scene::SceneIsect>
KERNEL_LAUNCH d_render_block_regeneration(float4* d_output, Scene scene) {
  // dynamically allocated with blockDim.y
  __shared__ uint head_block;
  __shared__ uint n_idle_in_block;

  if (threadIdx.x == 0) {
    n_idle_in_block = blockDim.x;
  }

  // persistent thread id
  int tid = threadIdx.x + (blockDim.x) * blockIdx.x;
  bool active = false;

  uint image_id;
  Path path;
  Isect isect;
  Rng rng(seed + tid);

  // persistent thread loop
  do {
    __syncthreads();

    if (paths_head_global >= c_n_paths && !active) {
      return;
    }

    // check if all the threads in a warp are idle
    if (n_idle_in_block >= blockDim.x) {
      //---------------REGENERATE--------------------
      __syncthreads();

      // let first thread get the head for all the block
      if (threadIdx.x == 0) {
        head_block = atomicAdd(&paths_head_global, blockDim.x);
        n_idle_in_block = 0;
      }

      __syncthreads();

      // path index per thread in a warp
      uint path_id = head_block + threadIdx.x;

      if (path_id >= c_n_paths) {
        return;
      }

      image_id = (path_id % (uint)(c_resolution.x * c_resolution.y));

      float2 pixel_index;
      pixel_index.x = (float)(image_id % ((uint)c_resolution.x)) + c_offset.x;
      pixel_index.y = (floorf((float)image_id / c_resolution.x)) + c_offset.y;

      path.ray = indexToCameraRay(pixel_index, c_pixel_index_range,
                                  c_raster_to_view, c_inv_view_mat, rng);
      path.throughput = make_float4(1.f, 1.f, 1.f, 1.f);

      active = true;
    }

    if (active) {
#ifdef RAYS_STATISTICS
      atomicAdd(&d_n_rays_statistics, 1);
#endif
      if (!scene.intersect(path.ray.o, path.ray.d, isect)) {
        atomicVectorAdd(
            &d_output[image_id],
            path.throughput * scene.Le(path.ray.o, path.ray.d, isect));
        atomicAdd(&n_idle_in_block, 1);
        active = false;
      } else {
        // intersecting volume BB
        float sampled_distance;
        auto* medium = scene.getMedium(isect);

        if (medium == 0 ||
            !medium->sampleDistance(path.ray.o, path.ray.d, isect.dist, rng,
                                    sampled_distance)) {
          // outside volume
          Frame frame;
          frame.setFromZ(isect.normal);
          float3 dir = frame.toLocal(normalize(-path.ray.d));

          path.ray.o = path.ray.o + path.ray.d * isect.dist;
          float weight = 1;

          if (scene.getBsdf(isect).sample(dir, path.ray.d, weight, rng)) {
            path.throughput *= weight;
            path.ray.d = frame.toWorld(path.ray.d);
            path.ray.o = path.ray.o + path.ray.d * EPSILON;
          }
        } else {
          path.ray.o =
              path.ray.o + path.ray.d * sampled_distance - path.ray.d * EPSILON;

          float4 albedo = medium->sampleAlbedo(path.ray.o);
          path.throughput = path.throughput * albedo;
          path.ray.d = medium->samplePhase(path.ray.d, rng);
        }
      }  // intersected medium

#ifdef RUSSIAN_ROULETTE
      /*
       *------roulette----
       */
      float pSurvive = fmin(1.f, fmaxf3(path.throughput));
      if (rng.getFloat() > pSurvive) {
        atomicAdd(&n_idle_in_block, 1);
        active = false;
      }
      path.throughput = path.throughput * 1.f / pSurvive;
#endif
    }

  } while (true);
}

};  // namespace RegenerationVolPTsk_kernel
#endif