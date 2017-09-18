#ifndef RNG_H_
#define RNG_H_

#include <cuda_runtime.h>
#include <helper_math.h>

#define CURAND

#ifdef CURAND

#include <curand.h>
#include <curand_kernel.h>

class Rng {
 public:
  typedef curandState State;

  __host__ __device__ Rng(State state) { state_ = state; }

  __host__ __device__ State getState() { return state_; }

  __device__ Rng(int seed = 1234) { curand_init(seed, 0, 0, &state_); }

  __host__ __device__ float getFloat() {
#ifdef __CUDA_ARCH__
    return curand_uniform(&state_);
#else
    return 0;
#endif
  }

  __host__ __device__ uint getUint() {
#ifdef __CUDA_ARCH__
    return (uint)curand_uniform(&state_);
#else
    return 0;
#endif
  }

  __host__ __device__ float2 getFloat2() {
    float a = getFloat();
    float b = getFloat();

    return make_float2(a, b);
  }

  __host__ __device__ float3 getFloat3() {
    float a = getFloat();
    float b = getFloat();
    float c = getFloat();

    return make_float3(a, b, c);
  }

 private:
  State state_;
};

#else

// thrust
#include <thrust/random.h>
class Rng {
 public:
  __host__ __device__ Rng(int seed = 1234)
      : rng_(seed), uniform_real_dist_(0, 1) {}

  __host__ __device__ float getFloat() { return uniform_real_dist_(rng_); }

  __host__ __device__ uint getUint() { return rng_(); }

  __host__ __device__ float2 getFloat2() {
    float a = getFloat();
    float b = getFloat();

    return make_float2(a, b);
  }

  __host__ __device__ float3 getFloat3() {
    float a = getFloat();
    float b = getFloat();
    float c = getFloat();

    return make_float3(a, b, c);
  }

 private:
  thrust::default_random_engine rng_;
  thrust::uniform_real_distribution<float> uniform_real_dist_;
};
#endif

#endif