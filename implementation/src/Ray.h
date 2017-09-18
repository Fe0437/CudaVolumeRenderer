/*
 * UtilMathStructs.h
 *
 *  Created on: 05/set/2017
 *      Author: macbook
 */

#ifndef RAY_H_
#define RAY_H_

// 1. CUDA headers first
#include <cuda_runtime.h>

// 4. Project headers
#include "CVRMath.h"
#include "Defines.h"
#include "Rng.h"

struct Ray {
  float3 o{};  // origin
  float3 d{};  // direction
};

struct Rays {
  float3* o{};  // origin
  float3* d{};  // direction
};

struct Path {
  Ray ray{};
  float4 throughput{1, 1, 1, 1};
};

struct Paths {
  Rays rays{};
  float4* throughputs{};
};

template <int ELEMENTS_PER_THREAD>
struct CubThread {
  float3 ray_o[ELEMENTS_PER_THREAD]{};  // origin
  float3 ray_d[ELEMENTS_PER_THREAD]{};  // direction
  float4 throughput[ELEMENTS_PER_THREAD]{};
  uint image_id[ELEMENTS_PER_THREAD]{};
  bool active[ELEMENTS_PER_THREAD]{};
};

struct Thread {
  Path path{};
  uint image_id{};
  bool active{};
};

struct Threads {
  Paths paths{};
  uint* image_ids{};
};

// doesn't need SoA because is usally read from registers
struct SimpleIsect {
  __host__ __device__ SimpleIsect() {}

  __host__ __device__ SimpleIsect(float max_dist) : dist(max_dist) {}

  float dist{};     //!< Distance to closest intersection (serves as ray.tmax)
  float3 normal{};  //!< Normal at the intersection
  bool inside_volume{};
};

#endif /* RAY_H_ */
