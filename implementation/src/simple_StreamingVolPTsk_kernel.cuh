#ifndef SIMPLE_STREAMING_VOLPT_SK_KERNEL_H_
#define SIMPLE_STREAMING_VOLPT_SK_KERNEL_H_
#pragma once

#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include "Bsdf.h"
#include "CVRMath.h"
#include "Geometry.h"
#include "Intersect.h"
#include "MortonSort.h"
#include "Ray.h"
#include "Rng.h"
#include "Utilities.cuh"
#include "Volume.h"
#include "helper_cuda.h"

namespace simple_StreamingVolPTsk_kernel {

typedef DeviceMedium Medium;

enum Variant { kClassic, kSortingRays };

__constant__ uint c_seed;
__constant__ uint c_n_paths;

__device__ uint d_n_active[STREAMING_SK_GRID_DIM_X] = {0, 0};
__device__ uint d_paths_head_global = 0;

template <int BLOCK_THREADS, int ELEMENTS_PER_THREAD, class BSDF,
          Variant VARIANT = kSortingRays>
struct BlockStreamingVolPT {
  typedef cub::BlockLoad<float3, BLOCK_THREADS, ELEMENTS_PER_THREAD,
                         cub::BLOCK_LOAD_WARP_TRANSPOSE>
      BlockLoadFloat3T;
  typedef cub::BlockLoad<float4, BLOCK_THREADS, ELEMENTS_PER_THREAD,
                         cub::BLOCK_LOAD_WARP_TRANSPOSE>
      BlockLoadFloat4T;
  typedef cub::BlockLoad<uint, BLOCK_THREADS, ELEMENTS_PER_THREAD,
                         cub::BLOCK_LOAD_WARP_TRANSPOSE>
      BlockLoadUintT;
  typedef cub::BlockLoad<bool, BLOCK_THREADS, ELEMENTS_PER_THREAD,
                         cub::BLOCK_LOAD_WARP_TRANSPOSE>
      BlockLoadBoolT;
  typedef cub::BlockScan<int, BLOCK_THREADS,
                         cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING>
      BlockScanT;
  typedef BlockMortonSort<BLOCK_THREADS, ELEMENTS_PER_THREAD, int>
      BlockMortonSortT;
  typedef cub::BlockReduce<int, BLOCK_THREADS> BlockReduceT;

  struct Load {
    BlockLoadFloat3T::TempStorage tfloat3;
    BlockLoadFloat4T::TempStorage tfloat4;
    BlockLoadUintT::TempStorage tuint;
  };

  union TempStorage {
    typename Load load;
    typename BlockScanT::TempStorage scan;
    typename BlockMortonSortT::TempStorage sort;
    typename BlockReduceT::TempStorage reduce;
  };

  TempStorage& temp_storage;

  __device__ __forceinline__ BlockStreamingVolPT(TempStorage& _temp_storage)
      : temp_storage(_temp_storage) {}

  __device__ __forceinline__ void regenerate(Thread& thread, Threads d_threads,
                                             Rng& rng) {
    //---------------REGENERATE--------------------
    if (threadIdx.x >= d_n_active[blockIdx.x]) {
      if (d_paths_head_global >= c_n_paths) {
        thread.active = false;
        return;
      }

      // get next path
      // uint path_id = atomicAggInc(&d_paths_head_global);
      uint path_id = atomicAdd(&d_paths_head_global, 1);

      if (path_id >= c_n_paths) {
        thread.active = false;
        return;
      }

      uint img_id = path_id % (uint)(c_resolution.x * c_resolution.y);

      float2 pixel_index;
      pixel_index.x = (float)(img_id % ((uint)c_resolution.x)) + c_offset.x;
      pixel_index.y = (floorf((float)img_id / c_resolution.x)) + c_offset.y;

      rng = Rng(c_seed + path_id);

      thread.path.ray = indexToCameraRay(pixel_index, c_pixel_index_range,
                                         c_raster_to_view, c_inv_view_mat, rng);
      thread.path.throughput = make_float4(1, 1, 1, 1);
      thread.image_id = img_id;

      d_threads.paths.rays.o[threadIdx.x] = thread.path.ray.o;
      d_threads.paths.rays.d[threadIdx.x] = thread.path.ray.d;
      d_threads.paths.throughputs[threadIdx.x] = thread.path.throughput;
      d_threads.image_ids[threadIdx.x] = thread.image_id;
    }

    thread.active = true;
  }

  __device__ __forceinline__ void scanAndCompact(
      Thread (&_thread)[ELEMENTS_PER_THREAD], Threads& d_threads) {
    auto& thread = _thread[0];
    int active_position = (int)thread.active;
    int aggregate = 0;

    BlockScanT(temp_storage.scan)
        .ExclusiveSum(active_position, active_position, aggregate);

    __syncthreads();

    __shared__ int start_position;

    if (threadIdx.x == 0) {
      start_position = atomicAdd(&d_n_active[blockIdx.x], aggregate);
    }

    __syncthreads();

    if (thread.active) {
      int index = start_position + active_position;
      d_threads.paths.rays.o[index] = thread.path.ray.o;
      d_threads.paths.rays.d[index] = thread.path.ray.d;
      d_threads.paths.throughputs[index] = thread.path.throughput;
      d_threads.image_ids[index] = thread.image_id;
    }
  }

  template <Variant VARIANT>
  __device__ __forceinline__ void compact(Medium& medium,
                                          Thread (&thread)[ELEMENTS_PER_THREAD],
                                          Threads& d_threads);

  template <>
  __device__ __forceinline__ void compact<kClassic>(
      Medium& medium, Thread (&thread)[ELEMENTS_PER_THREAD],
      Threads& d_threads) {
    scanAndCompact(thread, d_threads);
  }

  template <>
  __device__ __forceinline__ void compact<kSortingRays>(
      Medium& medium, Thread (&thread)[ELEMENTS_PER_THREAD],
      Threads& d_threads) {
    int index[1];
    index[0] = threadIdx.x;

    int aggregate =
        BlockReduceT(temp_storage.reduce).Sum((int)thread[0].active);

    d_threads.paths.rays.o[index[0]] = thread[0].path.ray.o;
    d_threads.paths.rays.d[index[0]] = thread[0].path.ray.d;
    d_threads.paths.throughputs[index[0]] = thread[0].path.throughput;
    d_threads.image_ids[index[0]] = thread[0].image_id;

    // sort all the threads in the block
    BlockMortonSortT(temp_storage.sort)
        .mortonSort(thread[0].path.ray.o, medium.density_AABB, index,
                    thread[0].active);

    // store the thread to move in a local variabe
    thread[0].path.ray.o = d_threads.paths.rays.o[index[0]];
    thread[0].path.ray.d = d_threads.paths.rays.d[index[0]];
    thread[0].path.throughput = d_threads.paths.throughputs[index[0]];
    thread[0].image_id = d_threads.image_ids[index[0]];

    __syncthreads();

    // sum all the active threads
    if (threadIdx.x == 0) {
      atomicAdd(&d_n_active[blockIdx.x], aggregate);
    }

    // move the thread
    d_threads.paths.rays.o[threadIdx.x] = thread[0].path.ray.o;
    d_threads.paths.rays.d[threadIdx.x] = thread[0].path.ray.d;
    d_threads.paths.throughputs[threadIdx.x] = thread[0].path.throughput;
    d_threads.image_ids[threadIdx.x] = thread[0].image_id;
  }

  // n_paths = 0 && d_threads_in is full
  __device__ __forceinline__ void extend(Thread (&thread)[ELEMENTS_PER_THREAD],
                                         Threads d_threads, float4* d_output,
                                         Medium medium, BSDF bsdf, Rng& rng) {
    auto& _thread = thread[0];
    // aliases
    float3& ray_o = _thread.path.ray.o;
    float3& ray_d = _thread.path.ray.d;
    float4& throughput = _thread.path.throughput;
    uint& image_id = _thread.image_id;
    bool& active = _thread.active;

    // check
    if (active) {
      // while paths are not regenerated anymore and thread is still active
      do {
        Isect isect;

        if (!medium.density_AABB.intersect(ray_o, ray_d, isect)) {
          atomicAdd(&d_output[image_id], throughput);
          active = false;
        } else {
          // intersecting volume BB
          float sampled_distance;

          if (!isect.inside_volume ||
              !medium.sampleDistance(ray_o, ray_d, isect.dist, rng,
                                     sampled_distance)) {
            // outside volume
            Frame frame;
            frame.setFromZ(isect.normal);
            float3 dir = frame.toLocal(normalize(-ray_d));

            ray_o = ray_o + ray_d * isect.dist;
            float weight = 1;

            if (bsdf.sample(dir, ray_d, weight, rng)) {
              throughput *= weight;
              ray_d = frame.toWorld(ray_d);
              ray_o = ray_o + ray_d * EPSILON;
            }
          } else {
            ray_o = ray_o + ray_d * sampled_distance - ray_d * EPSILON;

            float3 coord = worldToGrid(ray_o, medium.density_AABB.getExtent());
            float4 albedo =
                tex3D<float4>(medium.albedo_volume.volume_tex, int(coord.x),
                              int(coord.y), int(coord.z));
            throughput = throughput * albedo;
            ray_d = medium.phase.sample(ray_d, rng);
          }
        }  // intersected medium
      } while (d_paths_head_global > c_n_paths &&
               active);  // while paths are not regenerated anymore and thread
                         // is still active
    }  // if active check

    compact<VARIANT>(medium, thread, d_threads);
  }

  __device__ __forceinline__ void offsetThreads(Threads& d_threads,
                                                int offset) {
    d_threads.paths.rays.o = d_threads.paths.rays.o + offset;
    d_threads.paths.rays.d = d_threads.paths.rays.d + offset;
    d_threads.paths.throughputs = d_threads.paths.throughputs + offset;
    d_threads.image_ids = d_threads.image_ids + offset;
  }

  __device__ __forceinline__ void loadThread(Thread& thread,
                                             Threads d_threads) {
    // Per-thread tile data
    float3 t_ray_o[ELEMENTS_PER_THREAD];
    float3 t_ray_d[ELEMENTS_PER_THREAD];
    float4 t_throughput[ELEMENTS_PER_THREAD];
    uint t_image_id[ELEMENTS_PER_THREAD];

    // Load items into a blocked arrangement
    BlockLoadFloat3T(temp_storage.load.tfloat3)
        .Load(d_threads.paths.rays.o, t_ray_o);
    BlockLoadFloat3T(temp_storage.load.tfloat3)
        .Load(d_threads.paths.rays.d, t_ray_d);
    BlockLoadFloat4T(temp_storage.load.tfloat4)
        .Load(d_threads.paths.throughputs, t_throughput);
    BlockLoadUintT(temp_storage.load.tuint)
        .Load(d_threads.image_ids, t_image_id);

    thread.path.ray.o = t_ray_o[0];
    thread.path.ray.d = t_ray_d[0];
    thread.path.throughput = t_throughput[0];
    thread.image_id = t_image_id[0];
  }
};

//=================================KERNEL
//CALL===========================================

template <int BLOCK_THREADS, typename BSDF, Variant VARIANT = kClassic>
__global__ void d_render(Threads d_threads, float4* d_output, Medium medium,
                         BSDF bsdf) {
  typedef BlockStreamingVolPT<BLOCK_THREADS, 1, BSDF, VARIANT>
      BlockStreamingVolPT_T;

  __shared__ typename BlockStreamingVolPT_T::TempStorage temp_storage;
  BlockStreamingVolPT_T volpt(temp_storage);

  int offset = blockDim.x * blockIdx.x;
  volpt.offsetThreads(d_threads, offset);
  // 1 Random generator
  Rng rng;
  Thread thread[1];

  do {
    // 1 load for iteration
    volpt.loadThread(thread[0], d_threads);
    // regenerate the inactive threads
    volpt.regenerate(thread[0], d_threads, rng);

    __syncthreads();
    // reset the active threads count
    d_n_active[blockIdx.x] = 0;
    // extend the current thread rays
    volpt.extend(thread, d_threads, d_output, medium, bsdf, rng);

    __syncthreads();

  } while (d_n_active[blockIdx.x] > 0 || d_paths_head_global < c_n_paths);
}
}  // namespace simple_StreamingVolPTsk_kernel
#endif
