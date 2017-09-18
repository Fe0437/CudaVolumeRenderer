/*
 * CudaVolPath.h
 *
 *  Created on: 30/ago/2017
 *      Author: macbook
 */
#pragma once

#ifndef CUDAVOLPATH_H_
#define CUDAVOLPATH_H_

#include <cuda_runtime.h>
#include <driver_types.h>

#include <iterator>
#include <numeric>

#include "AbstractRenderer.h"
#include "CVRMath.h"
#include "Config.h"
#include "Debug.h"
#include "Defines.h"
#include "Ray.h"
#include "RenderKernelLauncher.h"
#include "Rng.h"
#include "Utilities.h"
#include "helper_cuda.h"
#include "helper_math.h"

namespace UtilityFunctors {
struct Scale;
}

template <class VolPathKernelLauncher>
class CudaVolPath : public AbstractProgressiveRenderer {
 public:
  using OutputDelegate = Buffer2DTransferDelegate<UtilityFunctors::Scale>;

  struct VolumeDataPointer {
    cudaArray* cuda_array = 0;
    void* cuda_malloc = nullptr;
  };

  using TileArray = std::vector<uint2>;

 private:
  // configurations
  PathTracingConfig path_tracing_config_{};
  TilingConfig tiling_config_{};
  CudaConfig cuda_config_{};

  Scene scene_{};
  uint current_iteration_{};

  TileArray::iterator current_tile_{};
  TileArray tiles_{};

  cudaEvent_t current_buffer_ready_event_{}, processing_buffer_ready_event_{};

  float4* d_output_{};
  float4* d_output_processed_{};

  VolumeDataPointer d_albedo_{};
  VolumeDataPointer d_density_{};

  std::unique_ptr<OutputDelegate> output_delegate_{};
  VolPathKernelLauncher kernel_launcher_{};

  std::vector<cudaTextureObject_t> volume_textures_{};

 public:
  CudaVolPath(const Config& config, std::unique_ptr<OutputDelegate>&&);
  ~CudaVolPath() override;

  void setOutputBufferTransferDelegate(
      std::unique_ptr<OutputDelegate>&& output_delegate) {
    output_delegate_ = std::move(output_delegate);
  }

  // progressive rendering interface
  void initRendering() override;
  void runIterations() override;
  void setNIterations(uint iterations) override;
  bool imageComplete() override;
  void getImage(Buffer2D buffer_out) override;

  // rendering interface
  void render(Buffer2D buffer_out) override;

 private:
  template <class VolumeType>
  cudaTextureObject_t createTextureWithVolume(VolumeDataPointer d_array,
                                              VolumeType* h_volume,
                                              cudaExtent volumeSize);
  void initCamera();
  void initRenderState();
  void prepareForNextIterations();
  void allocateDeviceMemory();
  void initDeviceScene();
  void releaseDeviceMemory();
  void initTileArray();
};

// Simple counting iterator
template <typename T>
class counting_iterator {
  T value;

 public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = T;
  using difference_type = T;
  using pointer = T*;
  using reference = T&;

  explicit counting_iterator(T start = 0) : value(start) {}
  auto operator*() const -> T { return value; }
  auto operator++() -> counting_iterator& {
    ++value;
    return *this;
  }
  auto operator++(int) -> counting_iterator {
    counting_iterator tmp(*this);
    ++value;
    return tmp;
  }
  auto operator==(const counting_iterator& other) const -> bool {
    return value == other.value;
  }
  auto operator!=(const counting_iterator& other) const -> bool {
    return value != other.value;
  }
};

#endif /* CUDAVOLPATH_H_ */
