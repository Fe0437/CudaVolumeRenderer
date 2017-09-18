#ifndef BSDF_H_
#define BSDF_H_

#include "GGX.h"

struct BSDF {
  __host__ __device__ virtual inline bool sample(const float3& input_dir,
                                                 float3& output_dir,
                                                 float& weight,
                                                 Rng& rng) const {
    output_dir = input_dir * -1;
    weight = 1.f;
    return true;
  }
};

struct GGX {
  float2 roughness{0.1f, 0.1f};
  float int_ior_over_ext_ior{};

  __host__ __device__ GGX(float int_ior = 1.05, float ext_ior = 1.01)
      : int_ior_over_ext_ior(int_ior / ext_ior) {}

  __host__ __device__ inline bool sample(const float3& input_dir,
                                         float3& output_dir, float& weight,
                                         Rng& rng) const {
    return GGX_sample(roughness, int_ior_over_ext_ior, input_dir, &rng,
                      &output_dir, &weight);
  }
};

#endif  // BSDF_H_
