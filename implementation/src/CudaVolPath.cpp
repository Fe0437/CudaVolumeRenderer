/*
 * CudaVolPath.cpp
 *
 *  Created on: 30/ago/2017
 *      Author: Federico
 */

#include "CudaVolPath.h"

CUDAVOLPATH_TEMPLATES

template <class VolPathKernelLauncher>
void CudaVolPath<VolPathKernelLauncher>::initTileArray() {
  tiles_ = TileArray(tiling_config_.n_tiles.x * tiling_config_.n_tiles.y);

  auto transform = [&](uint id) -> uint2 {
    return make_uint2(
        tiling_config_.tile_dim.x * (id % tiling_config_.n_tiles.x),
        tiling_config_.tile_dim.y * int((float)id / tiling_config_.n_tiles.x));
  };

  for (TileArray::iterator iter = tiles_.begin(); iter != tiles_.end();
       ++iter) {
    int tile_id = iter - tiles_.begin();
    *iter = transform(tile_id);
  }

  current_tile_ = tiles_.begin();
}

template <class VolPathKernelLauncher>
CudaVolPath<VolPathKernelLauncher>::CudaVolPath(
    const Config& config, std::unique_ptr<OutputDelegate>&& output_delegate)
    : scene_(config.scene),
      path_tracing_config_(config.path_tracing_config),
      tiling_config_(config.tiling_config),
      cuda_config_(config.cuda_config),
      output_delegate_(std::move(output_delegate)) {
  initTileArray();

  kernel_launcher_.copyRasterToView(scene_.getCamera()->getRasterToView());

  cudaEventCreateWithFlags(&current_buffer_ready_event_,
                           cudaEventDisableTiming);

#ifdef DOUBLE_BUFFERING
  cudaEventCreateWithFlags(&processing_buffer_ready_event_,
                           cudaEventDisableTiming);
#endif

  kernel_launcher_.setCudaConfig(config.cuda_config);
  kernel_launcher_.setResolution(tiling_config_.tile_dim);
  kernel_launcher_.copyPixelIndexRange(
      make_float2(tiling_config_.resolution.x, tiling_config_.resolution.y));
  kernel_launcher_.init();

  allocateDeviceMemory();
  initDeviceScene();
}

template <class VolPathKernelLauncher>
CudaVolPath<VolPathKernelLauncher>::~CudaVolPath() {
  releaseDeviceMemory();
}

template <class VolPathKernelLauncher>
void CudaVolPath<VolPathKernelLauncher>::initCamera() {
  const float* model_view_mat = scene_.getCamera()->getModelViewMatrix();

  // cuda samples
  float inv_view_mat[12];
  inv_view_mat[0] = model_view_mat[0];
  inv_view_mat[1] = model_view_mat[4];
  inv_view_mat[2] = model_view_mat[8];
  inv_view_mat[3] = model_view_mat[12];
  inv_view_mat[4] = model_view_mat[1];
  inv_view_mat[5] = model_view_mat[5];
  inv_view_mat[6] = model_view_mat[9];
  inv_view_mat[7] = model_view_mat[13];
  inv_view_mat[8] = model_view_mat[2];
  inv_view_mat[9] = model_view_mat[6];
  inv_view_mat[10] = model_view_mat[10];
  inv_view_mat[11] = model_view_mat[14];
  kernel_launcher_.copyInvViewMatrix(inv_view_mat, sizeof(float4) * 3);
}

template <class VolPathKernelLauncher>
void CudaVolPath<VolPathKernelLauncher>::initDeviceScene() {
  // TODO: more c++ style

  auto medium = scene_.getMedium();

  auto albedo_volume = medium.albedo_volume;
  auto density_volume = medium.density_volume;

  typename VolPathKernelLauncher::DeviceScene device_scene;
  auto& device_medium = device_scene.medium;

  device_medium.albedo_volume.volume_tex = createTextureWithVolume<float4>(
      d_albedo_, albedo_volume.getVolumeData(), albedo_volume.getCudaExtent());
  device_medium.albedo_volume.grid_resolution =
      medium.albedo_volume.grid_resolution;

  device_medium.density_volume.volume_tex =
      createTextureWithVolume<float>(d_density_, density_volume.getVolumeData(),
                                     density_volume.getCudaExtent());
  device_medium.density_volume.grid_resolution =
      medium.density_volume.grid_resolution;

  device_medium.max_density = medium.max_density;
  device_medium.scale = medium.scale;
  device_medium.density_AABB = medium.density_AABB;

  kernel_launcher_.setScene(device_scene);
}

template <class VolPathKernelLauncher>
template <class VolumeType>
cudaTextureObject_t CudaVolPath<VolPathKernelLauncher>::createTextureWithVolume(
    VolumeDataPointer volume_data, VolumeType* h_volume,
    cudaExtent volumeSize) {
  cudaTextureObject_t texture = 0;
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  if (cuda_config_.unified_memory) {  // not enough global memory for the
                                      // texture

    // TODO with pascal is possible to use unified memory access which are paged
    if (cuda_config_.device_properties.canMapHostMemory) {
      size_t bytes = volumeSize.width * volumeSize.height * volumeSize.depth *
                     (sizeof(VolumeType));
      checkCudaErrors(cudaHostAlloc((void**)&volume_data.cuda_malloc, bytes,
                                    cudaHostAllocMapped));
      cudaMemcpy(volume_data.cuda_malloc, h_volume, bytes,
                 cudaMemcpyHostToHost);

      texRes.resType = cudaResourceTypeLinear;
      texRes.res.linear.devPtr = volume_data.cuda_malloc;
      texRes.res.linear.desc.f = cudaChannelFormatKindFloat;
      texRes.res.linear.desc.x = 32;  // bits per channel
      texRes.res.linear.sizeInBytes = bytes;
    } else {
      throw std::exception(" not enough global memory avaiable ");
    }
  } else {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
    // create 3D array
    checkCudaErrors(
        cudaMalloc3DArray(&volume_data.cuda_array, &channelDesc, volumeSize));

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr =
        make_cudaPitchedPtr(h_volume, volumeSize.width * sizeof(VolumeType),
                            volumeSize.width, volumeSize.height);
    copyParams.dstArray = volume_data.cuda_array;
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = volume_data.cuda_array;
  }

  CHECK_CUDA_ERROR("allocated 3DArray");

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));
  texDescr.normalizedCoords = false;
#ifdef MITSUBA_COMPARABLE
  texDescr.filterMode = cudaFilterModePoint;
#else
  texDescr.filterMode = cudaFilterModeLinear;
#endif
  texDescr.addressMode[0] = cudaAddressModeClamp;  // clamp
  texDescr.addressMode[1] = cudaAddressModeClamp;
  texDescr.addressMode[2] = cudaAddressModeClamp;
  texDescr.readMode = cudaReadModeElementType;

  cudaCreateTextureObject(&texture, &texRes, &texDescr, NULL);
  CHECK_CUDA_ERROR("created 3D texture object");

  volume_textures_.push_back(texture);
  return texture;
}

template <class VolPathKernelLauncher>
void CudaVolPath<VolPathKernelLauncher>::prepareForNextIterations() {
  kernel_launcher_.reset();

  cudaEventSynchronize(processing_buffer_ready_event_);

  if (tiles_.size() !=
      1) {  // with only 1 tile the buffer is used for accumulate the result
    int tot_pixels = tiling_config_.tile_dim.x * tiling_config_.tile_dim.y;
    checkCudaErrors(
        cudaMemset(d_output_processed_, 0, tot_pixels * sizeof(float4)));
  }
}

template <class VolPathKernelLauncher>
void CudaVolPath<VolPathKernelLauncher>::initRenderState() {
  current_iteration_ = 0;
  int tot_pixels = tiling_config_.tile_dim.x * tiling_config_.tile_dim.y;
  checkCudaErrors(
      cudaMemset(d_output_processed_, 0, tot_pixels * sizeof(float4)));
  cudaEventRecord(processing_buffer_ready_event_);
}

template <class VolPathKernelLauncher>
void CudaVolPath<VolPathKernelLauncher>::allocateDeviceMemory() {
  int tot_pixels = tiling_config_.tile_dim.x * tiling_config_.tile_dim.y;
  checkCudaErrors(cudaMalloc((void**)&d_output_, tot_pixels * sizeof(float4)));

#ifdef DOUBLE_BUFFERING
  if (tiles_.size() != 1) {
    checkCudaErrors(
        cudaMalloc((void**)&d_output_processed_, tot_pixels * sizeof(float4)));
    else {
      d_output_processed_ = d_output_;
      processing_buffer_ready_event_ = current_buffer_ready_event_;
    }
#else
  d_output_processed_ = d_output_;
  processing_buffer_ready_event_ = current_buffer_ready_event_;
#endif

    kernel_launcher_.setOutputPtr(d_output_processed_);
    kernel_launcher_.allocateDeviceMemory();
    CHECK_CUDA_ERROR("kernel launcher allocated memory ");
  }

  template <class VolPathKernelLauncher>
  void CudaVolPath<VolPathKernelLauncher>::setNIterations(uint iterations) {
    path_tracing_config_.iterations = iterations;
    kernel_launcher_.setNIterations(path_tracing_config_.iterations);
  }

  template <class VolPathKernelLauncher>
  void CudaVolPath<VolPathKernelLauncher>::initRendering() {
    initCamera();
    initRenderState();
  }

  // render next tile at the end swap the buffer for processing (next time will
  // use another one)
  template <class VolPathKernelLauncher>
  void CudaVolPath<VolPathKernelLauncher>::runIterations() {
    // Reset iterator if needed
    if (current_tile_ == tiles_.end()) {
      current_tile_ = tiles_.begin();
    }

    if (current_tile_ == tiles_.begin()) {
      current_iteration_ += kernel_launcher_.getNIterations();
      LOG_DEBUG("iterations %d \n", current_iteration_)
    }

    uint2 tile_start = *current_tile_;
    kernel_launcher_.copyOffset(tile_start);

    kernel_launcher_.launchRender();

    LOG_DEBUG_IF(tiles_.size() != 1, "tile :  %d ,  %d  \n",
                 (int)(tile_start.x / tiling_config_.tile_dim.x),
                 (int)(tile_start.y / tiling_config_.tile_dim.y))

    ++current_tile_;

#ifdef DOUBLE_BUFFERING

    if (tiles_.size() != 1) {
      std::swap(d_output_, d_output_processed_);
      std::swap(processing_buffer_ready_event_, current_buffer_ready_event_);
      kernel_launcher_.setOutputPtr(d_output_processed_);
    }

#endif
  }

  template <class VolPathKernelLauncher>
  void CudaVolPath<VolPathKernelLauncher>::getImage(Buffer2D buffer_out) {
    assert(output_delegate_);

    Buffer2D buffer_in = make_buffer2D<float4>(
        d_output_, tiling_config_.tile_dim.x, tiling_config_.tile_dim.y);

    uint2 tile_start = *(current_tile_ - 1);

    output_delegate_->transfer(buffer_in, buffer_out, tile_start,
                               UtilityFunctors::Scale(current_iteration_));
    cudaEventRecord(current_buffer_ready_event_);
    prepareForNextIterations();
  }

  template <class VolPathKernelLauncher>
  void CudaVolPath<VolPathKernelLauncher>::releaseDeviceMemory() {
    cudaDeviceSynchronize();
    checkCudaErrors(cudaFree(d_output_));
    checkCudaErrors(cudaEventDestroy(current_buffer_ready_event_));

#ifdef DOUBLE_BUFFERING

    if (tiles_.size() != 1) {
      checkCudaErrors(cudaFree(d_output_processed_));
      checkCudaErrors(cudaEventDestroy(processing_buffer_ready_event_));
    }

#endif

    if (d_albedo_.cuda_array != 0) {
      checkCudaErrors(cudaFreeArray(d_albedo_.cuda_array));
    }
    if (d_density_.cuda_array != 0) {
      checkCudaErrors(cudaFreeArray(d_density_.cuda_array));
    }
    if (d_albedo_.cuda_malloc != 0) {
      checkCudaErrors(cudaFree(d_albedo_.cuda_malloc));
    }
    if (d_density_.cuda_malloc != 0) {
      checkCudaErrors(cudaFree(d_density_.cuda_malloc));
    }

    for (auto txt = volume_textures_.begin(); txt != volume_textures_.end();
         ++txt) {
      cudaDestroyTextureObject(*txt);
    }

    kernel_launcher_.releaseDeviceMemory();
  }

  template <class VolPathKernelLauncher>
  bool CudaVolPath<VolPathKernelLauncher>::imageComplete() {
    return current_tile_ == tiles_.end();
  }

  template <class VolPathKernelLauncher>
  void CudaVolPath<VolPathKernelLauncher>::render(Buffer2D buffer_out) {
    setNIterations(path_tracing_config_.iterations);
    initRendering();

    while (!imageComplete()) {
      runIterations();
      getImage(buffer_out);
    }
  }
