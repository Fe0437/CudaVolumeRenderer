#ifndef NAIVE_VOLPT_MK_KERNEL_H_
#define NAIVE_VOLPT_MK_KERNEL_H_

#include <cuda_runtime.h>

#include "Bsdf.h"
#include "CVRMath.h"
#include "Medium.h"
#include "Ray.h"
#include "Rng.h"
#include "Utilities.cuh"
#include "helper_cuda.h"

namespace NaiveVolPTmk_kernel {

#ifndef NAIVE_MK_COMPACTION
__device__ int d_n_active = 0;
#endif

template <typename Scene, typename Isect = typename Scene::SceneIsect>
KERNEL_LAUNCH d_init(Path* d_paths, int* d_active_pixels, float4* d_output,
                     Scene scene, const int current_iteration) {
  float2 pos;
  pos.x = blockIdx.x * blockDim.x + threadIdx.x;
  pos.y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((pos.x >= c_resolution.x) || (pos.y >= c_resolution.y)) return;

  float2 pixel_index = pos + c_offset;
  uint img_id = pos.x + (c_resolution.x * pos.y);

  Rng rng = makeSeededRng(current_iteration, img_id, 0);
  Path path;
  path.ray = indexToCameraRay(pixel_index, c_pixel_index_range,
                              c_raster_to_view, c_inv_view_mat, rng);

#ifdef RAYS_STATISTICS
  atomicAdd(&d_n_rays_statistics, 1);
#endif

  // intersection should be done by Scene
  Isect isect;
  bool hit = scene.intersect(path.ray.o, path.ray.d, isect);

  if (!hit) {
    d_active_pixels[img_id] = -1;
    d_output[img_id] += make_float4(1, 1, 1, 1);
    return;
  }

  if (isect.dist < 0.0f) {
    isect.dist = 0.0f;  // clamp to near plane
  }

  path.ray.o = path.ray.o + path.ray.d * isect.dist;

  float weight = 1;
  Frame frame;
  frame.setFromZ(isect.normal);

  float3 dir = frame.toLocal(normalize(-path.ray.d));
  if (scene.getBsdf(isect).sample(dir, path.ray.d, weight, rng)) {
    path.throughput *= weight;
    path.ray.d = frame.toWorld(path.ray.d);
    path.ray.o = path.ray.o + path.ray.d * EPSILON;
  } else {
    d_active_pixels[img_id] = -1;
    return;
  }

#ifndef NAIVE_MK_COMPACTION
  atomicAdd(&d_n_active, 1);
#endif

  d_active_pixels[img_id] = img_id;
  d_paths[img_id] = path;
}

template <typename Scene, typename Isect = typename Scene::SceneIsect>
KERNEL_LAUNCH d_extend(Path* d_paths, int* d_activePixels,
                       const int n_active_pixels, float4* d_output, Scene scene,
                       const int current_iteration, const int depth) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_active_pixels) return;

  int img_id = d_activePixels[tid];
  if (img_id == -1) return;

  Path path = d_paths[img_id];
  Rng rng = makeSeededRng(current_iteration, img_id, depth);
  float3 e = rng.getFloat3();

#ifdef RAYS_STATISTICS
  atomicAdd(&d_n_rays_statistics, 1);
#endif

  Isect isect;
  bool hit = scene.intersect(path.ray.o, path.ray.d, isect);

  if (!hit) {
    // not intersecting volume BB
    atomicVectorAdd(&d_output[img_id], path.throughput);
    d_activePixels[tid] = -1;
    return;
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
  }

#ifdef RUSSIAN_ROULETTE
  /*
   *------roulette----
   */
  float pSurvive = fmin(1.f, fmaxf3(path.throughput));
  if (rng.getFloat() > pSurvive) {
    d_activePixels[tid] = -1;
    return;
  }
  path.throughput = path.throughput * 1.f / pSurvive;
#endif

#ifndef NAIVE_MK_COMPACTION
  atomicAdd(&d_n_active, 1);
#endif
  d_paths[img_id] = path;
}
}  // namespace NaiveVolPTmk_kernel

#endif  //