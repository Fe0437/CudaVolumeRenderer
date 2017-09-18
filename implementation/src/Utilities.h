#pragma once
#include <cuda_runtime.h>

namespace UtilityFunctors {

struct Scale {
  float scale;
  __host__ __device__ Scale(float _scale) : scale(_scale) {}
  __host__ __device__ Scale(const Scale& functor) : scale(functor.scale) {}

  __host__ __device__ float operator()(float x) {
    const float value = x / scale;
    return value;
  }
};

struct IsNegative {
  __host__ __device__ bool operator()(const int& x) { return x < 0; }
};

}  // namespace UtilityFunctors

// https://en.wikipedia.org/wiki/Smoothstep
inline float smoothStep(float edge0, float edge1, float x) {
  // Scale, bias and saturate x to 0..1 range
  x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
  // Evaluate polynomial
  return x * x * (3 - 2 * x);
}

// code from
// https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__host__ __device__ __forceinline__ unsigned int expandBits(unsigned int v) {
  // arcane bit-swizzling properties of integer multiplication
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__host__ __device__ __forceinline__ unsigned int morton3D(float x, float y,
                                                          float z) {
  x = min(max(x * 1024.0f, 0.0f), 1023.0f);
  y = min(max(y * 1024.0f, 0.0f), 1023.0f);
  z = min(max(z * 1024.0f, 0.0f), 1023.0f);
  unsigned int xx = expandBits((unsigned int)x);
  unsigned int yy = expandBits((unsigned int)y);
  unsigned int zz = expandBits((unsigned int)z);
  return xx * 4 + yy * 2 + zz;
}

__host__ __device__ __forceinline__ unsigned int getMortonIndex(
    unsigned int code, double scale) {
  unsigned int index = (unsigned int)(double(code) * scale);
  return index;
}

__host__ __device__ __forceinline__ unsigned int getMortonIndex(
    int i, int range_i, int j, int range_j, int k, int range_k) {
  double scale_factor =
      double(((range_i) * (range_j) * (range_k))) / double(0x3FFFFFFFu);
  return getMortonIndex(
      morton3D((float)i / range_i, (float)j / range_j, (float)k / range_k),
      scale_factor);
}

inline void mortonIndexingTest() {
  unsigned int n = 4;
  unsigned int m = 4;
  unsigned int p = 4;

  unsigned int i, j, k;

  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      for (k = 0; k < p; k++) {
        unsigned int value = morton3D((float)i / n, (float)j / m, (float)k / p);
        double index = n * m * p * (double(value) / double(0x3FFFFFFFu));
        printf("value %d for position (%d %d %d) \n", (int)index, i, j, k);
      }
    }
  }
}