#ifndef IMAGE_BUFFER_TRANSFER_H_
#define IMAGE_BUFFER_TRANSFER_H_

#pragma once

#include <cuda_runtime.h>

#include "AbstractRenderer.h"
#include "Buffer.h"
#include "CVRMath.h"
#include "helper_cuda.h"

namespace UtilityFunctors {
struct Scale;
}

class HostImageBufferTansferDelegate
    : public Buffer2DTransferDelegate<UtilityFunctors::Scale> {
 public:
  void transfer(Buffer2D buffer_in, Buffer2D buffer_out, uint2 offset_out,
                UtilityFunctors::Scale transform) override;
};

class DeviceImageBufferTansferDelegate
    : public Buffer2DTransferDelegate<UtilityFunctors::Scale> {
 public:
  void transfer(Buffer2D buffer_in, Buffer2D buffer_out, uint2 offset_out,
                UtilityFunctors::Scale transform) override;
};

class DeviceTiledImageBufferTansferDelegate
    : public Buffer2DTransferDelegate<UtilityFunctors::Scale> {
  float4* d_buffer_transfer_ = 0;
  size_t d_buffer_bytes_;

 public:
  explicit DeviceTiledImageBufferTansferDelegate(size_t size)
      : d_buffer_bytes_(size) {
    checkCudaErrors(cudaMalloc(&d_buffer_transfer_, d_buffer_bytes_));
  }

  void transfer(Buffer2D buffer_in, Buffer2D buffer_out, uint2 offset_out,
                UtilityFunctors::Scale transform) override;
};

#endif  // !IMAGE_BUFFER_TRANSFER_H_
