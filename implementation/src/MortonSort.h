#pragma once

#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include "Geometry.h"
#include "Ray.h"
#include "Utilities.cuh"

template <int BLOCK_THREADS, int ELEMENTS_PER_THREAD, typename SORTED_VALUE_T>
struct BlockMortonSort {
  using uint = unsigned int;

  typedef cub::BlockRadixSort<uint, BLOCK_THREADS, ELEMENTS_PER_THREAD,
                              SORTED_VALUE_T>
      BlockRadixSortT;
  typedef BlockRadixSortT::TempStorage TempStorage;

  TempStorage& temp_storage;
  uint max{};

  __device__ __forceinline__ BlockMortonSort(TempStorage& _temp_storage)
      : temp_storage(_temp_storage) {
    max = morton3D(1, 1, 1);
  }

  __device__ __forceinline__ void mortonSort(
      float3 (&point)[ELEMENTS_PER_THREAD], AABB aligned_bounding_box,
      SORTED_VALUE_T (&value)[ELEMENTS_PER_THREAD],
      int filter[ELEMENTS_PER_THREAD]) {
    using namespace cub;

    uint code[ELEMENTS_PER_THREAD];
    aligned_bounding_box.transform(point);

#pragma unroll
    for (int ITEM = 0; ITEM < ELEMENTS_PER_THREAD; ITEM++) {
      if (filter[ITEM]) {
        code[ITEM] = morton3D(point[ITEM].x, point[ITEM].y, point[ITEM].z);
      } else {
        code[ITEM] = max;
      }
    }

    __syncthreads();

    BlockRadixSortT(temp_storage).Sort(code, value);
  }

  __device__ __forceinline__ void mortonSort(
      float3 (&point)[ELEMENTS_PER_THREAD], AABB aligned_bounding_box,
      SORTED_VALUE_T (&value)[ELEMENTS_PER_THREAD]) {
    using namespace cub;

    uint code[ELEMENTS_PER_THREAD];
    float3 points[ELEMENTS_PER_THREAD] = {point};
    aligned_bounding_box.transform(points);

#pragma unroll
    for (int ITEM = 0; ITEM < ELEMENTS_PER_THREAD; ITEM++) {
      code[ITEM] = morton3D(point[ITEM].x, point[ITEM].y, point[ITEM].z);
    }
    __syncthreads();

    BlockRadixSortT(temp_storage).Sort(code, value);
  }
};
