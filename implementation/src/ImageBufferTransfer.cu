#include "Buffer.h"
#include "ImageBufferTransfer.h"
#include "Occupancy.cuh"
#include "Utilities.h"

template <typename Out1T, typename Out4T, typename FUNC>
__global__ void d_transform(float4* buffer_in, Out4T* buffer_out,
                            unsigned int n, FUNC transform) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * 4) {
    // Get pointer to the specific byte we want to modify
    Out1T* out1_ptr = ((Out1T*)buffer_out) + (idx * sizeof(Out1T));
    float* float_ptr = ((float*)buffer_in) + idx;

    // Set only this specific value of the 4 values
    *out1_ptr = (Out1T)(transform(*float_ptr));
  }
}

template <typename Out4T, typename FUNC>
__global__ void d_accumulateAndTransform2D(
    float4* buffer_in, float4* buffer_transfer, Out4T* buffer_out,
    unsigned int input_width,   // Width of input buffer in pixels
    unsigned int output_width,  // Width of output buffer in pixels
    unsigned int height, uint2 offset, FUNC transform) {
  // Calculate pixel coordinates

  unsigned int local_x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int local_y = blockIdx.y * blockDim.y + threadIdx.y;

  // Calculate output coordinates with tile offset
  unsigned int out_x = local_x + offset.x;
  unsigned int out_y = local_y + offset.y;

  // Check bounds against image dimensions
  if (out_x >= output_width || out_y >= height) {
    return;
  }

  // Calculate indices using widths directly
  size_t index_in = local_y * input_width + local_x;
  size_t index_transfer = out_y * output_width + out_x;
  size_t index_out =
      index_transfer;  // Same as transfer since they share dimensions

  // Accumulate and transform
  float4 input = buffer_in[index_in];
  float4& transfer = buffer_transfer[index_transfer];
  transfer.x += input.x > 0 ? input.x : 0;
  transfer.y += input.y > 0 ? input.y : 0;
  transfer.z += input.z > 0 ? input.z : 0;

  // Transform and store result
  Out4T& out = buffer_out[index_out];
  out.x = transform(transfer.x);
  out.y = transform(transfer.y);
  out.z = transform(transfer.z);
  out.w = 255;
}

void HostImageBufferTansferDelegate::transfer(
    Buffer2D buffer_in, Buffer2D buffer_out, uint2 offset_out,
    UtilityFunctors::Scale transform) {
  size_t byte_offset =
      (offset_out.x * sizeof(uchar4)) + (offset_out.y * buffer_out.pitch_bytes);
  Buffer2D buffer_out_offset = buffer_out + byte_offset;

  dim3 block_size(32);
  uint n = buffer_out.width_bytes * buffer_out.height / sizeof(float);
  dim3 grid_size = dim3((n + block_size.x - 1) / block_size.x);

  d_transform<float><<<grid_size, block_size>>>(
      buffer_in.as<float4>(), buffer_in.as<float4>(), n, transform);
  cudaMemcpy2DAsync(buffer_out_offset.as<float4>(),
                    buffer_out_offset.pitch_bytes, buffer_in.as<float4>(),
                    buffer_in.pitch_bytes, buffer_out_offset.width_bytes,
                    buffer_out_offset.height, cudaMemcpyDeviceToHost);
}

template <typename FUNC>
struct ColorPixelTransform {
  FUNC transform;
  __host__ __device__ ColorPixelTransform(FUNC _transform)
      : transform(_transform) {}
  __host__ __device__ ColorPixelTransform(const ColorPixelTransform& copy)
      : transform(copy.transform) {}

  __host__ __device__ unsigned char operator()(float& pixel) {
    unsigned char c;
#ifdef __CUDA_ARCH__
    float gamma_corrected = __powf(transform(pixel), 1.f / 2.2f);
    c = clamp(gamma_corrected * 255.f, 0.f, 255.f);
#else
    float gamma_corrected = ::powf(transform(pixel), 1.f / 2.2f);
    float clamped = ::fminf(::fmaxf(gamma_corrected * 255.f, 0.f), 255.f);
    c = static_cast<unsigned char>(clamped);
#endif
    return c;
  }
};

void DeviceImageBufferTansferDelegate::transfer(
    Buffer2D buffer_in, Buffer2D buffer_out, uint2 offset_out,
    UtilityFunctors::Scale transform) {
  dim3 block_size(32);
  uint n = static_cast<uint>(
      static_cast<double>(buffer_out.width_bytes * buffer_out.height) /
      static_cast<double>(sizeof(unsigned char)));
  dim3 grid_size = dim3((n + block_size.x - 1) / block_size.x);

  d_transform<unsigned char><<<grid_size, block_size>>>(
      buffer_in.as<float4>(), buffer_out.as<uchar4>(), n,
      ColorPixelTransform<UtilityFunctors::Scale>(transform));
}

void DeviceTiledImageBufferTansferDelegate::transfer(
    Buffer2D buffer_in, Buffer2D buffer_out, uint2 offset_out,
    UtilityFunctors::Scale transform) {
  // Setup transfer buffer
  Buffer2D d_buffer_transfer = make_buffer2D<float4>(
      d_buffer_transfer_, buffer_out.width, buffer_out.height,
      d_buffer_bytes_ / buffer_out.height);

  // Calculate optimal grid configuration
  CudaConfig internal_cuda_config_;
  maxOccupancyGrid(internal_cuda_config_,
                   d_accumulateAndTransform2D<uchar4, UtilityFunctors::Scale>);

  // Calculate grid size using divUp and effective dimensions
  dim3 grid_size(divUp(buffer_in.width, internal_cuda_config_.block_size.x),
                 divUp(buffer_in.height, internal_cuda_config_.block_size.y));

  // Reset transfer buffer if needed
  if (transform.scale == 1) {
    cudaMemset2D(d_buffer_transfer.as<float4>(), d_buffer_transfer.pitch_bytes,
                 0, d_buffer_transfer.width_bytes, d_buffer_transfer.height);
  }

  // Launch kernel
  d_accumulateAndTransform2D<uchar4>
      <<<grid_size, internal_cuda_config_.block_size>>>(
          buffer_in.as<float4>(), d_buffer_transfer.as<float4>(),
          buffer_out.as<uchar4>(), buffer_in.width, buffer_out.width,
          buffer_out.height, offset_out,
          ColorPixelTransform<UtilityFunctors::Scale>(transform));
}