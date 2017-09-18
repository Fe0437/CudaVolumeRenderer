#ifndef NAIVE_VOLPT_SK_KERNEL_H_
#define NAIVE_VOLPT_SK_KERNEL_H_

#include <cuda_runtime.h>

#include "Bsdf.h"
#include "CVRMath.h"
#include "Geometry.h"
#include "Medium.h"
#include "Ray.h"
#include "Rng.h"
#include "Utilities.cuh"
#include "helper_cuda.h"

namespace NaiveVolPTsk_kernel {

template <typename Scene, typename Isect = typename Scene::SceneIsect>
KERNEL_LAUNCH d_render(float4* d_output, Scene scene) {
  int tid = threadIdx.x + (blockDim.x * blockIdx.x);
  if (tid >= c_n_paths) return;

  Rng rng(tid);
  uint image_id = (tid % (uint)(c_resolution.x * c_resolution.y));

  float2 pixel_index;
  pixel_index.x = (float)(image_id % ((uint)c_resolution.x)) + c_offset.x;
  pixel_index.y = (floorf((float)image_id / c_resolution.x)) + c_offset.y;

  Path path;
  path.ray = indexToCameraRay(pixel_index, c_pixel_index_range,
                              c_raster_to_view, c_inv_view_mat, rng);
  path.throughput = make_float4(1.f, 1.f, 1.f, 1.f);
  Isect isect;

  do {
#ifdef RAYS_STATISTICS
    atomicAdd(&d_n_rays_statistics, 1);
#endif

    if (!scene.intersect(path.ray.o, path.ray.d, isect)) {
      atomicVectorAdd(
          &d_output[image_id],
          path.throughput * scene.Le(path.ray.o, path.ray.d, isect));
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
    }  // intersected medium

#ifdef RUSSIAN_ROULETTE
    /*
     *------roulette----
     */
    float pSurvive = fmin(1.f, fmaxf3(path.throughput));
    if (rng.getFloat() > pSurvive) {
      return;
    }
    path.throughput = path.throughput * 1.f / pSurvive;
#endif

  } while (true);
}
}  // namespace NaiveVolPTsk_kernel
#endif