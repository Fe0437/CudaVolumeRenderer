#ifndef HG_H_
#define HG_H_
#pragma once

#include <cuda_runtime.h>

#include "CVRMath.h"
#include "Defines.h"

// generates local orthonormal basis around _dir
__host__ __device__ inline void generateLocalBasis(const float3 &_dir,
                                                   float3 &_vec1,
                                                   float3 &_vec2) {
  float invNorm1 = 1.0f / sqrtf(_dir.x * _dir.x + _dir.z * _dir.z);
  _vec1 = make_float3(_dir.z * invNorm1, 0.0f, -_dir.x * invNorm1);
  _vec2 = cross(_dir, _vec1);
}

__host__ __device__ inline float3 sphericalDirection(
    float _sinTheta, float _cosTheta, float _phi, const float3 &_x,
    const float3 &_y, const float3 &_z) {
  return _sinTheta * cosf(_phi) * _x + _sinTheta * sinf(_phi) * _y +
         _cosTheta * _z;
}

__host__ __device__ inline float PhaseHG(float _cosTheta, float _g) {
  return INV_FOURPI * (1.0f - _g * _g) /
         powf(1.0f + _g * _g - 2.0f * _g * _cosTheta, 1.5f);
}

__host__ __device__ inline float PhaseHG(const float3 &_vecIn, float3 &_vecOut,
                                         float _g) {
  float cosTheta = dot(_vecIn, _vecOut);
  return PhaseHG(cosTheta, _g);
}

__host__ __device__ inline float PdfHG(float _cosTheta, float _g) {
  return PhaseHG(_cosTheta, _g);
}

__host__ __device__ inline float PdfHG(const float3 &_vecIn, float3 &_vecOut,
                                       float _g) {
  return PhaseHG(_vecIn, _vecOut, _g);
}

__host__ __device__ inline float3 ImportanceSampleHG(const float3 &_v, float _g,
                                                     float e1, float e2) {
  float cosTheta;
  if (fabsf(_g) > EPSILON) {
    float sqrTerm = (1.0f - _g * _g) / (1.0f - _g + 2.0f * _g * e1);
    cosTheta = (1.0f + _g * _g - sqrTerm * sqrTerm) / (2.0f * fabsf(_g));
  } else {
    cosTheta = 1.0f - 2.0f * e1;
  }

  float sinTheta = sqrtf(max(0.0f, 1.0f - cosTheta * cosTheta));
  float phi = TWOPI * e2;

  float3 v1, v2;
  generateLocalBasis(_v, v1, v2);

  return sphericalDirection(sinTheta, cosTheta, phi, v1, v2, _v);
}

#endif