#pragma once
#include "Defines.h"

#define MIN_STEP 0.1
#define GRAD_DELTA_X make_float3(MIN_STEP, 0, 0)
#define GRAD_DELTA_Y make_float3(0, MIN_STEP, 0)
#define GRAD_DELTA_Z make_float3(0, 0, MIN_STEP)

template <class Volume>
__host__ __device__ inline float getVolumeIntensity(Volume& volume, float3& p) {
  if (p.x > 1 || p.y > 1 || p.z > 1 || p.x < 0 || p.y < 0 || p.z < 0) return 0;
  return volume(p);
}

template <class Volume>
__host__ __device__ inline float3 gradientCD(Volume& volume, float3& p) {
  const float Intensity[3][2] = {
      {getVolumeIntensity(volume, (p + GRAD_DELTA_X)),
       getVolumeIntensity(volume, (p - GRAD_DELTA_X))},
      {getVolumeIntensity(volume, (p + GRAD_DELTA_Y)),
       getVolumeIntensity(volume, (p - GRAD_DELTA_Y))},
      {getVolumeIntensity(volume, (p + GRAD_DELTA_Z)),
       getVolumeIntensity(volume, (p - GRAD_DELTA_Z))}};
  return make_float3(Intensity[0][1] - Intensity[0][0],
                     Intensity[1][1] - Intensity[1][0],
                     Intensity[2][1] - Intensity[2][0]);
}

template <class Volume>
__host__ __device__ inline float3 gradientFD(Volume& volume, float3& p) {
  const float Intensity[4] = {getVolumeIntensity(volume, p),
                              getVolumeIntensity(volume, (p + GRAD_DELTA_X)),
                              getVolumeIntensity(volume, (p + GRAD_DELTA_Y)),
                              getVolumeIntensity(volume, (p + GRAD_DELTA_Z))};

  return make_float3(Intensity[0] - Intensity[1], Intensity[0] - Intensity[2],
                     Intensity[0] - Intensity[3]);
}