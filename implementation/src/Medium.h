#pragma once

#include "Geometry.h"
#include "Gradient.h"
#include "Ray.h"
#include "Rng.h"
#include "Volume.h"

template <typename ALBEDO, typename DENSITY, typename PHASE>
class HeterogeneousMediumWithVariableBoundary {
 public:
  AABB density_AABB;
  float scale{};
  float max_density{};
  ALBEDO albedo_volume;
  DENSITY density_volume;
  PHASE phase;
  float density_threshold = 0.00000001;
  int seed = 1;

  __host__ __device__ HeterogeneousMediumWithVariableBoundary() {}

  __host__ __device__ HeterogeneousMediumWithVariableBoundary(
      const HeterogeneousMediumWithVariableBoundary& medium)
      : density_AABB(medium.density_AABB),
        scale(medium.scale),
        max_density(medium.max_density),
        albedo_volume(medium.albedo_volume),
        density_volume(medium.density_volume),
        phase(medium.phase) {}

  __host__ __device__ __forceinline__ float3 worldToVolume(float3 p) {
    density_AABB.transform(p);
    return p;
  }

  __device__ __forceinline__ bool sampleDistance(
      float3& ray_o, float3& ray_d, float& dist, Rng& rng,
      float& sampled_distance) const {
    sampled_distance = woodcockTracking(
        ray_o, ray_d, dist, max_density, density_volume,
        density_AABB.getExtent(), density_AABB.box_min, scale, &rng);
    return sampled_distance < dist;
  }

  __device__ __forceinline__ float4 sampleAlbedo(float3& ray_o) {
    float3 coord = worldToVolume(ray_o);
    return albedo_volume(coord);
  }

  __device__ __forceinline__ float3 samplePhase(float3& ray_d, Rng& rng) {
    return phase.sample(ray_d, rng);
  }

  template <typename Isect>
  __device__ __forceinline__ bool intersect(float3& ray_o, float3& ray_d,
                                            Isect& out_result) {
    bool hit = density_AABB.intersect(ray_o, ray_d, out_result);
    if (!hit) return false;

    Rng rng(++seed);
    const int max_iterations = 100000;
    int iter = 0;

    Isect temp_isect = out_result;
    float3 temp_o = ray_o + ray_d * temp_isect.dist + ray_d * EPSILON;

    float sign = 1;
    if (temp_isect.inside_volume) {
      sign = -1;
    }

    float3 temp_d = sign * ray_d;
    hit = density_AABB.intersect(temp_o, temp_d, temp_isect);
    if (!hit) return true;

    temp_o = temp_o - (MIN_STEP + EPSILON) * temp_d;
    float3 gradient = gradientCD(density_volume, worldToVolume(temp_o));

    float total_sampled_distance = 0;
    float new_dist = out_result.dist;
    while (norm3df(gradient.x, gradient.y, gradient.z) < density_threshold &&
           new_dist > 0) {
      if (++iter > max_iterations) return true;

      float sampled_distance = rng.getFloat() * MIN_STEP;
      total_sampled_distance += sampled_distance;
      new_dist = new_dist + sign * sampled_distance;

      if (new_dist < 0) return true;

      if (total_sampled_distance > temp_isect.dist) {
        out_result.inside_volume = !out_result.inside_volume;
        return false;
      }

      temp_o = temp_o + temp_d * sampled_distance;
      gradient = gradientCD(density_volume, worldToVolume(temp_o));
    }

    if (new_dist > 0) {
      out_result.dist = new_dist;
      if (total_sampled_distance != 0) out_result.normal = gradient;
    }

    return true;
  }
};

template <typename ALBEDO, typename DENSITY, typename PHASE>
class HeterogeneousMedium {
 public:
  AABB density_AABB;
  float scale{};
  float max_density{};
  ALBEDO albedo_volume;
  DENSITY density_volume;
  PHASE phase;

  __host__ __device__ HeterogeneousMedium() {}

  __host__ __device__ HeterogeneousMedium(const HeterogeneousMedium& medium)
      : density_AABB(medium.density_AABB),
        scale(medium.scale),
        max_density(medium.max_density),
        albedo_volume(medium.albedo_volume),
        density_volume(medium.density_volume),
        phase(medium.phase) {}

  __host__ __device__ __forceinline__ float3 worldToVolume(float3 p) {
    density_AABB.transform(p);
    return p;
  }

  __device__ __forceinline__ bool sampleDistance(
      float3& ray_o, float3& ray_d, float& dist, Rng& rng,
      float& sampled_distance) const {
    sampled_distance = woodcockTracking(
        ray_o, ray_d, dist, max_density, density_volume,
        density_AABB.getExtent(), density_AABB.box_min, scale, &rng);

    return sampled_distance < dist;
  }

  __device__ __forceinline__ float4 sampleAlbedo(float3& ray_o) {
    float3 coord = worldToVolume(ray_o);
    return albedo_volume(coord);
  }

  __device__ __forceinline__ float3 samplePhase(float3& ray_d, Rng& rng) {
    return phase.sample(ray_d, rng);
  }

  template <typename Isect>
  __device__ __forceinline__ bool intersect(float3& ray_o, float3& ray_d,
                                            Isect& out_result) {
    return density_AABB.intersect(ray_o, ray_d, out_result);
  }
};

template <class Medium, class Bsdf>
struct SimpleVolumeDeviceScene {
  using SceneIsect = SimpleIsect;

  SimpleVolumeDeviceScene() = default;
  Medium medium;
  Bsdf bsdf;

  __device__ __forceinline__ bool intersect(float3& ray_o, float3& ray_d,
                                            SceneIsect& out_result) {
    return medium.intersect(ray_o, ray_d, out_result);
  }

  __device__ __forceinline__ float4 Le(const float3& ray_o, const float3& ray_d,
                                       SceneIsect& out_result) {
    return make_float4(1.f);
  }

  __device__ __forceinline__ Medium* getMedium(SceneIsect& isect) {
    if (isect.inside_volume) {
      return &medium;
    }
    return 0;
  }

  __device__ __forceinline__ Bsdf& getBsdf(SceneIsect& isect) { return bsdf; }
};

typedef HeterogeneousMedium<DeviceVolume<float4>, DeviceVolume<float>, HG> DeviceMedium;
typedef HeterogeneousMedium<HostDeviceVolume<float4>, HostDeviceVolume<float>, HG> HostDeviceMedium;
typedef HeterogeneousMedium<Volume<float4>, Volume<float>, HG> HostMedium;