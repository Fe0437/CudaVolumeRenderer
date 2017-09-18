#pragma once

#include "CVRMath.h"
#include "Defines.h"
#include "Ray.h"

class AbstractGeometry {
 public:
  // Finds the closest intersection
  __host__ __device__ virtual bool intersect(const float3& ray_o,
                                             const float3& ray_d,
                                             SimpleIsect& out_result) = 0;

  // Finds any intersection, default calls Intersect
  __host__ __device__ virtual bool intersectP(const float3& ray_o,
                                              const float3& ray_d,
                                              SimpleIsect& out_result) {
    return intersect(ray_o, ray_d, out_result);
  }

  // Grows given bounding box by this object
  __host__ __device__ virtual void growBBox(float3& box_min,
                                            float3& box_max) = 0;
};

class AABB {
 public:
  float3 box_min{};
  float3 box_max{};

  __host__ __device__ AABB(const float3& _box_min, const float3& _box_max)
      : box_min(_box_min), box_max(_box_max) {}
  __host__ __device__ AABB(const AABB& aabb)
      : box_min(aabb.box_min), box_max(aabb.box_max) {}
  __host__ __device__ AABB() {}

  __host__ __device__ void growBBox(float3& _box_min, float3& _box_max) const {
    _box_min = box_min;
    _box_max = box_max;
  }

  __host__ __device__ float3 getExtent() const { return box_max - box_min; }

  template <int ITEMS = 1>
  __host__ __device__ void transform(float3 (&p)[ITEMS]) const {
    for (int ITEM = 0; ITEM < ITEMS; ITEM++) {
      transform(p[ITEM]);
    }
  }

  __host__ __device__ void transform(float3& p) const {
    p = (p - box_min) / (box_max - box_min);
  }

  template <typename ISECT = SimpleIsect>
  __host__ __device__ bool intersect(float3& ray_o, float3& ray_d,
                                     ISECT& out_result) const {
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / ray_d;
    float3 tbot = invR * (box_min - ray_o);
    float3 ttop = invR * (box_max - ray_o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    if (largest_tmin > EPSILON)
      out_result.dist = largest_tmin;
    else
      out_result.dist = smallest_tmax;

    if (out_result.dist == ttop.x)
      out_result.normal = make_float3(1, 0, 0);
    else if (out_result.dist == ttop.y)
      out_result.normal = make_float3(0, 1, 0);
    else if (out_result.dist == ttop.z)
      out_result.normal = make_float3(0, 0, 1);
    else if (out_result.dist == tbot.x)
      out_result.normal = make_float3(-1, 0, 0);
    else if (out_result.dist == tbot.y)
      out_result.normal = make_float3(0, -1, 0);
    else if (out_result.dist == tbot.z)
      out_result.normal = make_float3(0, 0, -1);

    out_result.inside_volume = dot(out_result.normal, ray_d) > 0;

    return (smallest_tmax > largest_tmin) && (out_result.dist > 0);
  }

  template <int ITEMS, typename ISECT = SimpleIsect>
  __host__ __device__ __forceinline__ void intersectBatch(
      float3 (&ray_o)[ITEMS], float3 (&ray_d)[ITEMS],
      ISECT (&out_result)[ITEMS], bool (&answer)[ITEMS]) const {
    for (int ITEM = 0; ITEM < ITEMS; ITEM++) {
      // compute intersection of ray with all six bbox planes
      float3 invR = make_float3(1.0f) / ray_d[ITEM];
      float3 tbot = invR * (box_min - ray_o[ITEM]);
      float3 ttop = invR * (box_max - ray_o[ITEM]);

      // re-order intersections to find smallest and largest on each axis
      float3 tmin = fminf(ttop, tbot);
      float3 tmax = fmaxf(ttop, tbot);

      // find the largest tmin and the smallest tmax
      float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
      float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

      if (largest_tmin > EPSILON)
        out_result[ITEM].dist = largest_tmin;
      else
        out_result[ITEM].dist = smallest_tmax;

      if (out_result[ITEM].dist == ttop.x)
        out_result[ITEM].normal = make_float3(1, 0, 0);
      else if (out_result[ITEM].dist == ttop.y)
        out_result[ITEM].normal = make_float3(0, 1, 0);
      else if (out_result[ITEM].dist == ttop.z)
        out_result[ITEM].normal = make_float3(0, 0, 1);
      else if (out_result[ITEM].dist == tbot.x)
        out_result[ITEM].normal = make_float3(-1, 0, 0);
      else if (out_result[ITEM].dist == tbot.y)
        out_result[ITEM].normal = make_float3(0, -1, 0);
      else if (out_result[ITEM].dist == tbot.z)
        out_result[ITEM].normal = make_float3(0, 0, -1);
      out_result[ITEM].inside_volume = dot(out_result[ITEM].normal, ray_d) > 0;

      answer[ITEM] =
          (smallest_tmax > largest_tmin) && (out_result[ITEM].dist > 0);
    }
  }
};