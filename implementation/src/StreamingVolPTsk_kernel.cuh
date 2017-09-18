#ifndef STREAMING_VOLPT_SK_KERNEL_H_
#define STREAMING_VOLPT_SK_KERNEL_H_
#pragma once

#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include "Bsdf.h"
#include "CVRMath.h"
#include "Geometry.h"
#include "Medium.h"
#include "MortonSort.h"
#include "Ray.h"
#include "Rng.h"
#include "Utilities.cuh"
#include "helper_cuda.h"

namespace StreamingVolPTsk_kernel {

enum Variant { kClassic, kSortingRays };

__constant__ uint c_seed;

__device__ uint d_paths_head_global = 0;

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, class Scene, class Isect,
          Variant VARIANT>
struct BlockStreamingVolPT {
  typedef cub::BlockLoad<float3, BLOCK_THREADS, ITEMS_PER_THREAD,
                         cub::BLOCK_LOAD_TRANSPOSE>
      BlockLoadFloat3T;
  typedef cub::BlockLoad<float4, BLOCK_THREADS, ITEMS_PER_THREAD,
                         cub::BLOCK_LOAD_TRANSPOSE>
      BlockLoadFloat4T;
  typedef cub::BlockLoad<uint, BLOCK_THREADS, ITEMS_PER_THREAD,
                         cub::BLOCK_LOAD_TRANSPOSE>
      BlockLoadUintT;

  typedef cub::BlockScan<int, BLOCK_THREADS,
                         cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING>
      BlockScanT;
  typedef BlockMortonSort<BLOCK_THREADS, ITEMS_PER_THREAD, int>
      BlockMortonSortT;
  typedef cub::BlockReduce<int, BLOCK_THREADS> BlockReduceT;

  union Load {
    BlockLoadFloat3T::TempStorage tfloat3_0;
    BlockLoadFloat3T::TempStorage tfloat3_1;
    BlockLoadFloat4T::TempStorage tfloat4;
    BlockLoadUintT::TempStorage tuint;
  };

  union TempStorage {
    typename Load load;
    typename BlockMortonSortT::TempStorage sort;
    typename BlockReduceT::TempStorage reduce;
    typename BlockScanT::TempStorage scan;
  };

  TempStorage& temp_storage;

  __device__ __forceinline__ BlockStreamingVolPT(TempStorage& _temp_storage)
      : temp_storage(_temp_storage) {}

  __device__ __forceinline__ void regenerate(Thread (&thread)[ITEMS_PER_THREAD],
                                             Threads& d_threads, int& n_active,
                                             Rng& rng) {
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
      thread[ITEM].active = true;

      //---------------REGENERATE--------------------
      if ((threadIdx.x * ITEMS_PER_THREAD) + ITEM >= n_active) {
        if (d_paths_head_global >= c_n_paths) {
          thread[ITEM].active = false;
          continue;
        }

        // get next path
        // path_id = atomicAggInc(&d_paths_head_global);
        uint path_id = atomicAdd(&d_paths_head_global, 1);

        if (path_id >= c_n_paths) {
          thread[ITEM].active = false;
          continue;
        }

        uint image_id = path_id % (uint)(c_resolution.x * c_resolution.y);

        float2 pixel_index;
        pixel_index.x = (float)(image_id % ((uint)c_resolution.x)) + c_offset.x;
        pixel_index.y = (floorf((float)image_id / c_resolution.x)) + c_offset.y;

        Ray ray = indexToCameraRay(pixel_index, c_pixel_index_range,
                                   c_raster_to_view, c_inv_view_mat, rng);

        thread[ITEM].path.ray.o = ray.o;
        thread[ITEM].path.ray.d = ray.d;
        thread[ITEM].path.throughput = make_float4(1, 1, 1, 1);
        thread[ITEM].image_id = image_id;
      }
    }
  }

  __device__ __forceinline__ void scatterThread(
      Thread (&thread)[ITEMS_PER_THREAD], Threads& d_threads,
      int(rank)[ITEMS_PER_THREAD]) {
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
      if (thread[ITEM].active) {
        d_threads.paths.rays.o[rank[ITEM]] = thread[ITEM].path.ray.o;
        d_threads.paths.rays.d[rank[ITEM]] = thread[ITEM].path.ray.d;
        d_threads.paths.throughputs[rank[ITEM]] = thread[ITEM].path.throughput;
        d_threads.image_ids[rank[ITEM]] = thread[ITEM].image_id;
      }
    }
  }

  __device__ __forceinline__ void swapThread(Thread (&thread)[ITEMS_PER_THREAD],
                                             Threads& d_threads,
                                             int rank[ITEMS_PER_THREAD]) {
    // store
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
      int id = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
      d_threads.paths.rays.o[id] = thread[ITEM].path.ray.o;
      d_threads.paths.rays.d[id] = thread[ITEM].path.ray.d;
      d_threads.paths.throughputs[id] = thread[ITEM].path.throughput;
      d_threads.image_ids[id] = thread[ITEM].image_id;
    }

    __syncthreads();

    // scatter
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
      thread[ITEM].path.ray.o = d_threads.paths.rays.o[rank[ITEM]];
      thread[ITEM].path.ray.d = d_threads.paths.rays.d[rank[ITEM]];
      thread[ITEM].path.throughput = d_threads.paths.throughputs[rank[ITEM]];
      thread[ITEM].image_id = d_threads.image_ids[rank[ITEM]];
    }

    __syncthreads();

    // store
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
      int id = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
      d_threads.paths.rays.o[id] = thread[ITEM].path.ray.o;
      d_threads.paths.rays.d[id] = thread[ITEM].path.ray.d;
      d_threads.paths.throughputs[id] = thread[ITEM].path.throughput;
      d_threads.image_ids[id] = thread[ITEM].image_id;
    }
  }

  __device__ __forceinline__ void scanAndCompact(
      Thread (&thread)[ITEMS_PER_THREAD], Threads& d_threads, int& n_active) {
    int items_id[ITEMS_PER_THREAD];

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
      items_id[ITEM] = (int)thread[ITEM].active;
    }

    int aggregate = 0;
    BlockScanT(temp_storage.scan).ExclusiveSum(items_id, items_id, aggregate);

    __syncthreads();

    if (threadIdx.x == 0) {
      n_active = aggregate;
    }

    scatterThread(thread, d_threads, items_id);
  }

  template <Variant VARIANT>
  __device__ __forceinline__ void compact(AABB& aabb,
                                          Thread (&thread)[ITEMS_PER_THREAD],
                                          Threads d_threads, int& n_active);

  template <>
  __device__ __forceinline__ void compact<kClassic>(
      AABB& aabb, Thread (&thread)[ITEMS_PER_THREAD], Threads d_threads,
      int& n_active) {
    scanAndCompact(thread, d_threads, n_active);
  }

  template <>
  __device__ __forceinline__ void compact<kSortingRays>(
      AABB& aabb, Thread (&thread)[ITEMS_PER_THREAD], Threads d_threads,
      int& n_active) {
    float3 t_ray_o[ITEMS_PER_THREAD];
    int t_active[ITEMS_PER_THREAD];
    int rank[ITEMS_PER_THREAD];

    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
      t_ray_o[ITEM] = thread[ITEM].path.ray.o;
      t_active[ITEM] = (int)thread[ITEM].active;
      rank[ITEM] = (ITEMS_PER_THREAD * threadIdx.x) + ITEM;
    }

    int aggregate = BlockReduceT(temp_storage.reduce).Sum(t_active);

    // sort all the threads in the block
    BlockMortonSortT(temp_storage.sort)
        .mortonSort(t_ray_o, aabb, rank, t_active);

    __syncthreads();

    // sum all the active threads
    if (threadIdx.x == 0) {
      n_active = aggregate;
    }

    swapThread(thread, d_threads, rank);
  }

  // n_paths = 0 && d_threads_in is full
  __device__ __forceinline__ void extend(Thread (&thread)[ITEMS_PER_THREAD],
                                         Threads d_threads, int& n_active,
                                         float4* d_output, Scene& scene,
                                         Rng& rng) {
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
        } while (d_paths_head_global > c_n_paths &&
                 active);  // while paths are not regenerated anymore and thread
                           // is still active
      }  // if active check
    }

    compact<VARIANT>(scene.medium.density_AABB, thread, d_threads, n_active);
  }

  __device__ __forceinline__ void offsetThreads(Threads& d_threads,
                                                int offset) {
    d_threads.paths.rays.o = d_threads.paths.rays.o + offset;
    d_threads.paths.rays.d = d_threads.paths.rays.d + offset;
    d_threads.paths.throughputs = d_threads.paths.throughputs + offset;
    d_threads.image_ids = d_threads.image_ids + offset;
  }

  __device__ __forceinline__ void loadThread(Thread (&thread)[ITEMS_PER_THREAD],
                                             Threads d_threads) {
    // Per-thread tile data
    float3 t_ray_o[ITEMS_PER_THREAD];
    float3 t_ray_d[ITEMS_PER_THREAD];
    float4 t_throughput[ITEMS_PER_THREAD];
    uint t_image_id[ITEMS_PER_THREAD];
    // bool t_active[ITEMS_PER_THREAD];

    BlockLoadFloat3T(temp_storage.load.tfloat3_0)
        .Load(d_threads.paths.rays.o, t_ray_o);
    __syncthreads();

    BlockLoadFloat3T(temp_storage.load.tfloat3_1)
        .Load(d_threads.paths.rays.d, t_ray_d);
    __syncthreads();

    BlockLoadFloat4T(temp_storage.load.tfloat4)
        .Load(d_threads.paths.throughputs, t_throughput);
    __syncthreads();

    BlockLoadUintT(temp_storage.load.tuint)
        .Load(d_threads.image_ids, t_image_id);
    __syncthreads();

    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
      thread[ITEM].path.ray.o = t_ray_o[ITEM];
      thread[ITEM].path.ray.d = t_ray_d[ITEM];
      thread[ITEM].path.throughput = t_throughput[ITEM];
      thread[ITEM].image_id = t_image_id[ITEM];
    }
  }

  __device__ __forceinline__ void render(Threads& d_threads, float4* d_output,
                                         Scene& scene) {
    int offset = blockDim.x * blockIdx.x * ITEMS_PER_THREAD;
    offsetThreads(d_threads, offset);

    __shared__ int n_active;
    if (threadIdx.x == 0) n_active = 0;

    Rng rng(c_seed + threadIdx.x + blockDim.x * blockIdx.x);
    Thread thread[ITEMS_PER_THREAD];

    do {
      // 1 load for iteration
      loadThread(thread, d_threads);

      // regenerate the inactive threads
      regenerate(thread, d_threads, n_active, rng);

      __syncthreads();

      // extend the current thread rays
      extend(thread, d_threads, n_active, d_output, scene, rng);

      __syncthreads();

    } while (n_active > 0 || d_paths_head_global < c_n_paths);
  }
};

//=================================KERNEL
//CALL===========================================

template <int BLOCK_THREADS, int ITEMS_PER_THREAD = 1, typename Scene,
          Variant VARIANT = kSortingRays>
KERNEL_LAUNCH d_render(Threads d_threads, float4* d_output, Scene scene) {
  typedef BlockStreamingVolPT<BLOCK_THREADS, ITEMS_PER_THREAD, Scene,
                              typename Scene::SceneIsect, VARIANT>
      BlockStreamingVolPT_T;
  __shared__ typename BlockStreamingVolPT_T::TempStorage temp_storage;

  BlockStreamingVolPT_T volpt(temp_storage);
  volpt.render(d_threads, d_output, scene);
}
}  // namespace StreamingVolPTsk_kernel
#endif
