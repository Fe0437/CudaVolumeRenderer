#ifndef CVR_MATH_H_
#define CVR_MATH_H_

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>

#include "Defines.h"

// cuda
#include <cuda_runtime.h>
#include <helper_math.h>

using float3x4 = struct {
  float4 m[3];
};

// transform vector by matrix (no translation)
__host__ __device__ inline float3 mul(const float3x4 &M, const float3 &v) {
  float3 r;
  r.x = dot(v, make_float3(M.m[0]));
  r.y = dot(v, make_float3(M.m[1]));
  r.z = dot(v, make_float3(M.m[2]));
  return r;
}

__host__ __device__ inline auto fmaxf3(float4 value) -> float {
  return fmaxf(fmaxf(value.x, value.y), value.z);
}

// transform vector by matrix with translation
__host__ __device__ inline float4 mul(const float3x4 &M, const float4 &v) {
  float4 r;
  r.x = dot(v, M.m[0]);
  r.y = dot(v, M.m[1]);
  r.z = dot(v, M.m[2]);
  r.w = 1.0F;
  return r;
}

inline auto divUp(int a, int b) -> int {
  return (a % b != 0) ? ((a / b) + 1) : (a / b);
}

inline __host__ __device__ float2 tan(float2 val) {
  return make_float2(tanf(val.x), tanf(val.y));
}

inline __host__ __device__ float2 operator+(float2 a, uint2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ float2 uint2ToFloat2(uint2 a) {
  return make_float2(a.x, a.y);
}

/// Coordinate frame
class Frame {
 public:
  __host__ __device__ Frame() {
    x_ = make_float3(1, 0, 0);
    y_ = make_float3(0, 1, 0);
    z_ = make_float3(0, 0, 1);
  };

  __host__ __device__ Frame(float3 x, float3 y, float3 z)
      : x_(x), y_(y), z_(z) {}

  __host__ __device__ void setFromZ(float3 z) {
    float3 tmpZ = z_ = normalize(z);
    float3 tmpX =
        (fabsf(tmpZ.x) > 0.99f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
    y_ = normalize(cross(tmpZ, tmpX));
    x_ = cross(y_, tmpZ);
  }

  __host__ __device__ float3 toWorld(float3 a) {
    return x_ * a.x + y_ * a.y + z_ * a.z;
  }

  __host__ __device__ float3 toLocal(float3 a) {
    return make_float3(dot(a, x_), dot(a, y_), dot(a, z_));
  }

  float3 Binormal() { return x_; }
  float3 Tangent() { return y_; }
  float3 Normal() { return z_; }

 public:
  float3 x_, y_, z_;
};

#endif CVR_MATH_H_