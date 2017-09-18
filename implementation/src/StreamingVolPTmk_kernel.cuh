#ifndef STREAMING_VOLPT_MK_KERNEL_H_
#define STREAMING_VOLPT_MK_KERNEL_H_
#pragma once

#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include "Bsdf.h"
#include "CVRMath.h"
#include "Geometry.h"
#include "Medium.h"
#include "Occupancy.cuh"
#include "Ray.h"
#include "Rng.h"
#include "Utilities.cuh"
#include "helper_cuda.h"

namespace StreamingVolPTmk_kernel {

__constant__ uint c_seed;

__device__ uint d_n_active = 0;
__device__ uint d_paths_head_global = 0;

template <int ITEMS_PER_THREAD = 1>
__global__ void d_regenerate(Threads d_threads_out, Rng::State* states,
                             bool* active_threads) {
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    uint item_id = ITEM + threadIdx.x * ITEMS_PER_THREAD +
                   blockDim.x * blockIdx.x * ITEMS_PER_THREAD;

    //---------------REGENERATE--------------------
    if (item_id >= d_n_active) {
      if (d_paths_head_global >= c_n_paths) {
        active_threads[item_id] = false;
        return;
      }

      // get next path
      uint path_id = atomicAdd(&d_paths_head_global, 1);

      if (path_id >= c_n_paths) {
        active_threads[item_id] = false;
        return;
      }

      uint img_id = path_id % (uint)(c_resolution.x * c_resolution.y);

      float2 pixel_index;
      pixel_index.x = (float)(img_id % ((uint)c_resolution.x)) + c_offset.x;
      pixel_index.y = (floorf((float)img_id / c_resolution.x)) + c_offset.y;

      Rng rng(c_seed + path_id);

      Path path;
      path.ray = indexToCameraRay(pixel_index, c_pixel_index_range,
                                  c_raster_to_view, c_inv_view_mat, rng);

      d_threads_out.paths.rays.o[item_id] = path.ray.o;
      d_threads_out.paths.rays.d[item_id] = path.ray.d;
      d_threads_out.paths.throughputs[item_id] = make_float4(1, 1, 1, 1);
      d_threads_out.image_ids[item_id] = img_id;
      active_threads[item_id] = true;
      states[threadIdx.x + blockDim.x * blockIdx.x] = rng.getState();
    }
  }
}

// n_paths = 0 && d_threads_in is full
template <int BLOCK_THREADS, int ITEMS_PER_THREAD = 1, class Scene,
          class Isect = typename Scene::SceneIsect>
KERNEL_LAUNCH d_extend(Threads d_threads_in, Threads d_threads_out,
                       float4* d_output, Scene scene, Rng::State* states,
                       bool* active_threads) {
  using namespace cub;

  typedef BlockLoad<float3, BLOCK_THREADS, ITEMS_PER_THREAD,
                    BLOCK_LOAD_TRANSPOSE>
      BlockLoadFloat3T;
  typedef BlockLoad<float4, BLOCK_THREADS, ITEMS_PER_THREAD,
                    BLOCK_LOAD_TRANSPOSE>
      BlockLoadFloat4T;
  typedef BlockLoad<uint, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE>
      BlockLoadUintT;
  typedef BlockLoad<bool, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE>
      BlockLoadBoolT;
  typedef BlockLoad<Rng::State, BLOCK_THREADS, 1, BLOCK_LOAD_TRANSPOSE>
      BlockLoadStateT;
  typedef BlockScan<int, BLOCK_THREADS, BlockScanAlgorithm::BLOCK_SCAN_RAKING>
      BlockScanT;

  Rng::State t_rnd_state[1];
  typename BlockLoadStateT::TempStorage load_state;
  BlockLoadStateT(load_state)
      .Load(states + blockDim.x * blockIdx.x, t_rnd_state);
  Rng rng(t_rnd_state[0]);

  int offset = blockDim.x * blockIdx.x * ITEMS_PER_THREAD;

  // Shared memory
  __shared__ union TempStorage {
    union Load {
      typename BlockLoadFloat3T::TempStorage tfloat3_0;
      typename BlockLoadFloat3T::TempStorage tfloat3_1;
      typename BlockLoadFloat4T::TempStorage tfloat4;
      typename BlockLoadBoolT::TempStorage tbool;
      typename BlockLoadUintT::TempStorage tuint;
    } load;

    typename BlockScanT::TempStorage scan;
  } temp_storage;

  Thread thread[ITEMS_PER_THREAD];

  {
    // Per-thread tile data
    float3 t_ray_o[ITEMS_PER_THREAD];
    float3 t_ray_d[ITEMS_PER_THREAD];
    float4 t_throughput[ITEMS_PER_THREAD];
    bool t_active[ITEMS_PER_THREAD];
    uint t_image_id[ITEMS_PER_THREAD];

    // Load items into a blocked arrangement
    BlockLoadFloat3T(temp_storage.load.tfloat3_0)
        .Load(d_threads_in.paths.rays.o + offset, t_ray_o);
    __syncthreads();
    BlockLoadFloat3T(temp_storage.load.tfloat3_1)
        .Load(d_threads_in.paths.rays.d + offset, t_ray_d);
    __syncthreads();
    BlockLoadFloat4T(temp_storage.load.tfloat4)
        .Load(d_threads_in.paths.throughputs + offset, t_throughput);
    __syncthreads();
    BlockLoadBoolT(temp_storage.load.tbool)
        .Load(active_threads + offset, t_active);
    __syncthreads();
    BlockLoadUintT(temp_storage.load.tuint)
        .Load(d_threads_in.image_ids + offset, t_image_id);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
      thread[ITEM].path.ray.o = t_ray_o[ITEM];
      thread[ITEM].path.ray.d = t_ray_d[ITEM];
      thread[ITEM].path.throughput = t_throughput[ITEM];
      thread[ITEM].image_id = t_image_id[ITEM];
      thread[ITEM].active = t_active[ITEM];
    }
  }

  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    // aliases
    float3& ray_o = thread[ITEM].path.ray.o;
    float3& ray_d = thread[ITEM].path.ray.d;
    float4& throughput = thread[ITEM].path.throughput;
    uint& image_id = thread[ITEM].image_id;
    bool& active = thread[ITEM].active;

    // check
    if (active) {
      // while paths are not regenerated anymore and thread is still active
      do {
        Isect isect;

#ifdef RAYS_STATISTICS
        atomicAdd(&d_n_rays_statistics, 1);
#endif

        if (!scene.intersect(ray_o, ray_d, isect)) {
          atomicVectorAdd(&d_output[image_id],
                          throughput * scene.Le(ray_o, ray_d, isect));
          active = false;
        } else {
          // intersecting volume BB
          float sampled_distance;
          auto* medium = scene.getMedium(isect);

          if (medium == 0 || !medium->sampleDistance(ray_o, ray_d, isect.dist,
                                                     rng, sampled_distance)) {
            // outside volume
            Frame frame;
            frame.setFromZ(isect.normal);
            float3 dir = frame.toLocal(normalize(-ray_d));

            ray_o = ray_o + ray_d * isect.dist;
            float weight = 1;

            if (scene.getBsdf(isect).sample(dir, ray_d, weight, rng)) {
              throughput *= weight;
              ray_d = frame.toWorld(ray_d);
              ray_o = ray_o + ray_d * EPSILON;
            }
          } else {
            ray_o = ray_o + ray_d * sampled_distance - ray_d * EPSILON;
            float4 albedo = medium->sampleAlbedo(ray_o);
            throughput = throughput * albedo;
            ray_d = medium->samplePhase(ray_d, rng);
          }
        }  // intersected medium

#ifdef RUSSIAN_ROULETTE
        /*
         *------roulette----
         */
        float pSurvive = fmin(1.f, fmaxf3(throughput));
        if (rng.getFloat() > pSurvive) {
          active = false;
        }
        throughput = throughput * 1.f / pSurvive;
#endif

      } while (d_paths_head_global >= c_n_paths &&
               active);  // while paths are not regenerated anymore and thread
                         // is still active
    }  // if active check
  }

  int items_id[ITEMS_PER_THREAD];

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    items_id[ITEM] = (int)thread[ITEM].active;
  }

  // synch for reusing of shared mem
  __syncthreads();

  int aggregate = 0;
  BlockScanT(temp_storage.scan).ExclusiveSum(items_id, items_id, aggregate);

  __syncthreads();

  __shared__ int start_position;

  if (threadIdx.x == 0) {
    start_position = atomicAdd(&d_n_active, aggregate);
  }

  __syncthreads();

  states[threadIdx.x + (blockDim.x * blockIdx.x)] = rng.getState();

  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (thread[ITEM].active) {
      items_id[ITEM] += start_position;
      d_threads_out.paths.rays.o[items_id[ITEM]] = thread[ITEM].path.ray.o;
      d_threads_out.paths.rays.d[items_id[ITEM]] = thread[ITEM].path.ray.d;
      d_threads_out.paths.throughputs[items_id[ITEM]] =
          thread[ITEM].path.throughput;
      d_threads_out.image_ids[items_id[ITEM]] = thread[ITEM].image_id;
      active_threads[items_id[ITEM]] = thread[ITEM].active;
    }
  }
}
}  // namespace StreamingVolPTmk_kernel
#endif
