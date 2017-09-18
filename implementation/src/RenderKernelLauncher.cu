#include "RenderKernelLauncher.h"

RENDER_KERNEL_LAUNCHER_TEMPLATES

template <typename VolumeType>
__device__ __forceinline__ VolumeType DeviceVolume<VolumeType>::get(uint x) {
  return tex1Dfetch<VolumeType>(volume_tex, x);
}

template <>
__device__ __forceinline__ float4 DeviceVolume<float4>::get(uint x) {
  float4 ret;
  ret.x = tex1Dfetch<float>(volume_tex, x * 4);
  ret.y = tex1Dfetch<float>(volume_tex, (x * 4) + 1);
  ret.z = tex1Dfetch<float>(volume_tex, (x * 4) + 2);
  ret.w = 1.f;
  return ret;
}

template <typename VolumeType>
__device__ __forceinline__ VolumeType DeviceVolume<VolumeType>::get(uint x,
                                                                    uint y,
                                                                    uint z) {
  return tex3D<VolumeType>(volume_tex, x, y, z);
}

template <typename VolumeType>
__device__ __forceinline__ VolumeType
HostDeviceVolume<VolumeType>::get(uint x, uint y, uint z) {
  return get((z * grid_resolution.x * grid_resolution.y) +
             (y * grid_resolution.x) + x);
}

template <typename VolumeType>
__device__ __forceinline__ VolumeType
HostDeviceVolume<VolumeType>::get(uint x) {
  return tex1Dfetch<VolumeType>(volume_tex, x);
}

template <>
__device__ __forceinline__ float4 HostDeviceVolume<float4>::get(uint x, uint y,
                                                                uint z) {
  float4 ret;
  ret.x = tex1Dfetch<float>(volume_tex,
                            (z * grid_resolution.x * grid_resolution.y * 4) +
                                (y * grid_resolution.x * 4) + x * 4);
  ret.y = tex1Dfetch<float>(volume_tex,
                            (z * grid_resolution.x * grid_resolution.y * 4) +
                                (y * grid_resolution.x * 4) + (x * 4) + 1);
  ret.z = tex1Dfetch<float>(volume_tex,
                            (z * grid_resolution.x * grid_resolution.y * 4) +
                                (y * grid_resolution.x * 4) + (x * 4) + 2);
  ret.w = 1.f;
  return ret;
}

template <>
__device__ __forceinline__ float4 HostDeviceVolume<float4>::get(uint x) {
  float4 ret;
  ret.x = tex1Dfetch<float>(volume_tex, x * 4);
  ret.y = tex1Dfetch<float>(volume_tex, (x * 4) + 1);
  ret.z = tex1Dfetch<float>(volume_tex, (x * 4) + 2);
  ret.w = 1.f;
  return ret;
}

__constant__ float3x4 c_inv_view_mat;  // inverse view matrix
__constant__ float2 c_raster_to_view;
__constant__ float2 c_resolution;
__constant__ uint2 c_offset;
__constant__ float2 c_pixel_index_range;
__constant__ uint c_n_paths;

#ifdef RAYS_STATISTICS
__device__ static int d_n_rays_statistics = 0;
extern int n_rays_traced_statistic;
#endif

#include "NaiveVolPTmk_kernel.cuh"
#include "NaiveVolPTsk_kernel.cuh"
#include "RegenerationVolPTsk_kernel.cuh"
#include "SortingVolPTsk_kernel.cuh"
#include "StreamingVolPTmk_kernel.cuh"
#include "StreamingVolPTsk_kernel.cuh"

void RenderKernelLauncher::copyInvViewMatrix(float* inv_view_mat,
                                             size_t size_of_mat) {
  cudaMemcpyToSymbol(c_inv_view_mat, inv_view_mat, size_of_mat);
}

void RenderKernelLauncher::copyRasterToView(float2 rtv) {
  cudaMemcpyToSymbol(c_raster_to_view, &rtv, sizeof(float2));
}

void RenderKernelLauncher::copyResolution(float2 resolution) {
  cudaMemcpyToSymbol(c_resolution, &resolution, sizeof(float2));
}

void RenderKernelLauncher::copyPixelIndexRange(float2 index_range) {
  cudaMemcpyToSymbol(c_pixel_index_range, &index_range, sizeof(float2));
}

void RenderKernelLauncher::copyOffset(uint2 offset) {
  cudaMemcpyToSymbol(c_offset, &offset, sizeof(uint2));
}

void RenderKernelLauncher::loadStatistics() {
#ifdef RAYS_STATISTICS
  cudaMemcpyToSymbol(d_n_rays_statistics, &n_rays_traced_statistic,
                     sizeof(int));
#endif
}

void RenderKernelLauncher::saveStatistics() {
#ifdef RAYS_STATISTICS
  CHECK_CUDA_ERROR("Error check after launching kernel : ");
  cudaMemcpyFromSymbol(&n_rays_traced_statistic, d_n_rays_statistics,
                       sizeof(int));
#endif
}

template <typename DeviceScene>
void VolPTKernelLauncher<DeviceScene>::setNIterations(uint n_iterations) {
  n_iterations_ = n_iterations;
  n_paths_ = resolution_.x * resolution_.y * n_iterations_;
  cudaMemcpyToSymbol(c_n_paths, &n_paths_, sizeof(uint));
}

//--------------------------------------NAIVE-SK----------------------------------------

template <typename DeviceScene>
void NaiveVolPTsk<DeviceScene>::init() {
  void* kernel = NaiveVolPTsk_kernel::d_render<DeviceScene>;

  // cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
  maxOccupancyConfig(cuda_config_, kernel, 0, 0);
  COUT_DEBUG("Cuda Config : <<< ("
             << cuda_config_.grid_size.x << "," << cuda_config_.grid_size.y
             << "," << cuda_config_.grid_size.z << ") , ("
             << cuda_config_.block_size.x << "," << cuda_config_.block_size.y
             << "," << cuda_config_.block_size.z << ") >>> \n")

  cuda_config_.dynamic_shared_memory = 0;
}

template <typename DeviceScene>
void NaiveVolPTsk<DeviceScene>::launchRender() {
  RenderKernelLauncher::loadStatistics();
  cuda_config_.grid_size =
      dim3(iDivUp(n_iterations_ * resolution_.x * resolution_.y,
                  cuda_config_.block_size.x));
  NaiveVolPTsk_kernel::
      d_render<<<cuda_config_.grid_size, cuda_config_.block_size,
                 cuda_config_.dynamic_shared_memory>>>(d_output_,
                                                       device_scene_);
  CHECK_CUDA_ERROR("render check ");
  RenderKernelLauncher::saveStatistics();
}

//---------------------------------------NAIVE-MK----------------------------------------

template <typename DeviceScene>
void NaiveVolPTmk<DeviceScene>::init() {
  void* kernel = NaiveVolPTmk_kernel::d_init<DeviceScene>;
  cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
}

template <typename DeviceScene>
void NaiveVolPTmk<DeviceScene>::allocateDeviceMemory() {
  n_active_pixels_ = resolution_.x * resolution_.y;
  checkCudaErrors(
      cudaMalloc(&d_active_pixels_, sizeof(int) * n_active_pixels_));
  checkCudaErrors(cudaMalloc(&d_paths_, sizeof(Path) * n_active_pixels_));
}

template <typename DeviceScene>
void NaiveVolPTmk<DeviceScene>::releaseDeviceMemory() {
  cudaFree(d_active_pixels_);
  cudaFree(d_paths_);
}

template <typename DeviceScene>
void NaiveVolPTmk<DeviceScene>::launchRender() {
  RenderKernelLauncher::loadStatistics();
  current_iteration_ = 0;

  void* kernel = NaiveVolPTmk_kernel::d_init<DeviceScene>;
  maxOccupancyConfig(cuda_config_, kernel, 0);

  cuda_config_.block_size.y = std::sqrt(cuda_config_.block_size.x);
  cuda_config_.block_size.x = cuda_config_.block_size.y;

  cuda_config_.grid_size =
      dim3((resolution_.x + cuda_config_.block_size.x - 1) /
               cuda_config_.block_size.x,
           (resolution_.y + cuda_config_.block_size.y - 1) /
               cuda_config_.block_size.y);

  COUT_DEBUG("Cuda Config : <<< ("
             << cuda_config_.grid_size.x << "," << cuda_config_.grid_size.y
             << "," << cuda_config_.grid_size.z << ") , ("
             << cuda_config_.block_size.x << "," << cuda_config_.block_size.y
             << "," << cuda_config_.block_size.z << ") >>> \n")

  for (int i = 0; i < n_iterations_; i++) {
    n_processed_pixels_ = n_active_pixels_;

    // LOG_DEBUG_IF(i % 100 == 0,
    // std::to_string(calculateOccupancy(cuda_config_,
    // NaiveVolPTmk_kernel::d_init<DeviceScene>)).c_str());

#ifndef NAIVE_MK_COMPACTION
    int zero = 0;
    cudaMemcpyToSymbol(NaiveVolPTmk_kernel::d_n_active, &zero, sizeof(int));
#endif

    NaiveVolPTmk_kernel::
        d_init<<<cuda_config_.grid_size, cuda_config_.block_size>>>(
            d_paths_, d_active_pixels_, d_output_, device_scene_,
            current_iteration_);

    // CHECK_CUDA_ERROR("INIT RAY");
#ifndef NAIVE_MK_COMPACTION
    cudaMemcpyFromSymbol(&n_processed_pixels_, NaiveVolPTmk_kernel::d_n_active,
                         sizeof(uint));
#endif
    uint bounce = 0;
    while (n_processed_pixels_ != 0) {
      extend(bounce, current_iteration_);
      ++bounce;
    }

    current_iteration_++;
    // LOG_DEBUG_IF( n_iterations_%100 == 0 , " completed : %f %\n",
    // ((float)current_iteration_ / (float)n_iterations_) * 100.f);
  }
  RenderKernelLauncher::saveStatistics();
}

template <typename DeviceScene>
void NaiveVolPTmk<DeviceScene>::extend(uint path_length,
                                       uint current_iteration) {
  CudaConfig extend_config;
  extend_config.block_size =
      cuda_config_.block_size.x * cuda_config_.block_size.y;
  extend_config.grid_size = dim3((int)std::fmaxf(
      1, (n_processed_pixels_ + extend_config.block_size.x - 1) /
             extend_config.block_size.x));

  // LOG_DEBUG_IF(current_iteration % 100 == 0,
  // std::to_string(calculateOccupancy(extend_config,
  // NaiveVolPTmk_kernel::d_extend<DeviceScene>)).c_str());

#ifndef NAIVE_MK_COMPACTION
  int zero = 0;
  cudaMemcpyToSymbol(NaiveVolPTmk_kernel::d_n_active, &zero, sizeof(int));
#endif

  NaiveVolPTmk_kernel::
      d_extend<<<extend_config.grid_size, extend_config.block_size>>>(
          d_paths_, d_active_pixels_, n_processed_pixels_, d_output_,
          device_scene_, current_iteration, path_length);

  CHECK_CUDA_ERROR("TRACE RAY");

#ifdef NAIVE_MK_COMPACTION
  thrust::device_ptr<int> device_pointer(d_active_pixels_);
  thrust::device_ptr<int> end =
      thrust::remove_if(device_pointer, device_pointer + n_processed_pixels_,
                        UtilityFunctors::IsNegative());
  n_processed_pixels_ = end.get() - d_active_pixels_ - 1;
#else
  cudaMemcpyFromSymbol(&n_processed_pixels_, NaiveVolPTmk_kernel::d_n_active,
                       sizeof(int));
#endif
}

//---------------------------------------REGENERATION-SK----------------------------------------

template <typename DeviceScene>
void RegenerationVolPTsk<DeviceScene>::init() {
  // cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);

#if REGENERATION_SYNCHRONIZATION_LEVEL == 0
  /*single thread regeneration*/
  void* kernel =
      RegenerationVolPTsk_kernel::d_render_single_thread_regeneration<
          DeviceScene>;
  maxOccupancyConfig(cuda_config_, kernel, 0);
  cuda_config_.dynamic_shared_memory = 0;

#elif REGENERATION_SYNCHRONIZATION_LEVEL == 1
  // warp regeneration
  void* kernel = RegenerationVolPTsk_kernel::d_render<DeviceScene>;

#if CUDART_VERSION < 9000
  float warp_size = cuda_config_.device_properties.warpSize;
  uint max = std::numeric_limits<int>::max();
  int limit = (n_paths_ < max ? n_paths_ : 0);
  maxOccupancyConfig(cuda_config_, kernel,
                     UtilityFunctors::Scale(warp_size / sizeof(uint)), limit);
  cuda_config_.block_size =
      dim3(warp_size, (float)cuda_config_.block_size.x / warp_size, 1);
#else

  maxOccupancyConfig(cuda_config_, kernel, 0);
  cuda_config_.dynamic_shared_memory = 0;
#endif
#else
  // block regeneration
  void* kernel =
      RegenerationVolPTsk_kernel::d_render_block_regeneration<DeviceScene>;
  maxOccupancyConfig(cuda_config_, kernel, 0);
  cuda_config_.dynamic_shared_memory = 0;

#endif

  COUT_DEBUG("Cuda Config : <<< ("
             << cuda_config_.grid_size.x << "," << cuda_config_.grid_size.y
             << "," << cuda_config_.grid_size.z << ") , ("
             << cuda_config_.block_size.x << "," << cuda_config_.block_size.y
             << "," << cuda_config_.block_size.z << ") >>> \n")
  COUT_DEBUG("Occupancy d_render : " << calculateOccupancy(cuda_config_)
                                     << "% \n")
}

template <typename DeviceScene>
void RegenerationVolPTsk<DeviceScene>::launchRender() {
  RenderKernelLauncher::loadStatistics();

#if REGENERATION_SYNCHRONIZATION_LEVEL == 0
  // single thread regeneration
  RegenerationVolPTsk_kernel::d_render_single_thread_regeneration<<<
      cuda_config_.grid_size, cuda_config_.block_size,
      cuda_config_.dynamic_shared_memory>>>(d_output_, device_scene_);
#elif REGENERATION_SYNCHRONIZATION_LEVEL == 1
  // warp regeneration
  RegenerationVolPTsk_kernel::
      d_render<<<cuda_config_.grid_size, cuda_config_.block_size,
                 cuda_config_.dynamic_shared_memory>>>(d_output_,
                                                       device_scene_);
#else
  // block regeneration
  RegenerationVolPTsk_kernel::d_render_block_regeneration<<<
      cuda_config_.grid_size, cuda_config_.block_size,
      cuda_config_.dynamic_shared_memory>>>(d_output_, device_scene_);
#endif

  // CHECK_CUDA_ERROR("render check ");
  RenderKernelLauncher::saveStatistics();
}

template <typename DeviceScene>
void RegenerationVolPTsk<DeviceScene>::reset() {
  RenderKernelLauncher::reset();
  uint zero = 0;
  cudaMemcpyToSymbol(RegenerationVolPTsk_kernel::paths_head_global, &zero,
                     sizeof(uint));
  seed_ += n_paths_;
  cudaMemcpyToSymbol(RegenerationVolPTsk_kernel::seed, &seed_, sizeof(uint));
}

//---------------------------------------STREAMING-MK-----------------------------------------------

template <typename DeviceScene>
void StreamingVolPTmk<DeviceScene>::init() {
  // Get suggested block size from CUDA API
  int minGridSize;
  int blockSize;
  void* kernel =
      StreamingVolPTmk_kernel::d_extend<STREAMING_THREADS_BLOCK,
                                        STREAMING_ITEMS_PER_THREAD, DeviceScene,
                                        typename DeviceScene::SceneIsect>;
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel);

  COUT_DEBUG("CUDA suggested block size for StreamingVolPTmk::d_extend: "
             << blockSize);
  COUT_DEBUG("CUDA suggested minimum grid size for StreamingVolPTmk::d_extend: "
             << minGridSize);

  cuda_config_.block_size.x = STREAMING_THREADS_BLOCK;
  maxOccupancyGrid(cuda_config_, kernel);

  COUT_DEBUG("Cuda Config : <<< ("
             << cuda_config_.grid_size.x << "," << cuda_config_.grid_size.y
             << "," << cuda_config_.grid_size.z << ") , ("
             << cuda_config_.block_size.x << "," << cuda_config_.block_size.y
             << "," << cuda_config_.block_size.z << ") >>> \n")
  COUT_DEBUG("Occupancy d_extend : " << calculateOccupancy(cuda_config_)
                                     << "% \n")
  n_threads_ = cuda_config_.grid_size.x * cuda_config_.block_size.x;
}

template <typename DeviceScene>
void StreamingVolPTmk<DeviceScene>::allocateDeviceMemory() {
  std::vector<Threads*> threads_buffers{&d_threads_in_, &d_threads_out_};

  for (int i = 0; i < threads_buffers.size(); i++) {
    auto thread_buffer = threads_buffers[i];
    checkCudaErrors(
        cudaMalloc(&thread_buffer->paths.rays.o,
                   STREAMING_ITEMS_PER_THREAD * n_threads_ * sizeof(float3)));
    checkCudaErrors(
        cudaMalloc(&thread_buffer->paths.rays.d,
                   STREAMING_ITEMS_PER_THREAD * n_threads_ * sizeof(float3)));
    checkCudaErrors(
        cudaMalloc(&thread_buffer->paths.throughputs,
                   STREAMING_ITEMS_PER_THREAD * n_threads_ * sizeof(float4)));
    checkCudaErrors(
        cudaMalloc(&thread_buffer->image_ids,
                   STREAMING_ITEMS_PER_THREAD * n_threads_ * sizeof(uint)));
  }
  checkCudaErrors(cudaMalloc(&active_threads_, STREAMING_ITEMS_PER_THREAD *
                                                   n_threads_ * sizeof(bool)));
  checkCudaErrors(cudaMalloc(&states_, n_threads_ * sizeof(Rng::State)));
}

template <typename DeviceScene>
void StreamingVolPTmk<DeviceScene>::releaseDeviceMemory() {
  std::vector<Threads*> threads_buffers{&d_threads_in_, &d_threads_out_};

  for (int i = 0; i < threads_buffers.size(); i++) {
    auto thread_buffer = threads_buffers[i];
    checkCudaErrors(cudaFree(thread_buffer->paths.rays.o));
    checkCudaErrors(cudaFree(thread_buffer->paths.rays.d));
    checkCudaErrors(cudaFree(thread_buffer->paths.throughputs));
    checkCudaErrors(cudaFree(thread_buffer->image_ids));
  }

  checkCudaErrors(cudaFree(active_threads_));
  checkCudaErrors(cudaFree(states_));
}

template <typename DeviceScene>
void StreamingVolPTmk<DeviceScene>::launchRender() {
  RenderKernelLauncher::loadStatistics();
  uint n_active = n_threads_;
  uint paths_head_global = 0;

  while (n_active > 0 || paths_head_global < n_paths_) {
    StreamingVolPTmk_kernel::d_regenerate<STREAMING_ITEMS_PER_THREAD>
        <<<cuda_config_.grid_size, cuda_config_.block_size,
           cuda_config_.dynamic_shared_memory>>>(d_threads_out_, states_,
                                                 active_threads_);
    CHECK_CUDA_ERROR("regenerate check ");

    n_active = 0;
    cudaMemcpyToSymbol(StreamingVolPTmk_kernel::d_n_active, &n_active,
                       sizeof(uint));
    std::swap(d_threads_in_, d_threads_out_);

    StreamingVolPTmk_kernel::d_extend<STREAMING_THREADS_BLOCK,
                                      STREAMING_ITEMS_PER_THREAD, DeviceScene,
                                      typename DeviceScene::SceneIsect>
        <<<cuda_config_.grid_size, cuda_config_.block_size,
           cuda_config_.dynamic_shared_memory>>>(d_threads_in_, d_threads_out_,
                                                 d_output_, device_scene_,
                                                 states_, active_threads_);
    CHECK_CUDA_ERROR("render check ");

    cudaMemcpyFromSymbol(&n_active, StreamingVolPTmk_kernel::d_n_active,
                         sizeof(uint));
    cudaMemcpyFromSymbol(&paths_head_global,
                         StreamingVolPTmk_kernel::d_paths_head_global,
                         sizeof(uint));

    LOG_DEBUG_IF((int)paths_head_global % 10000 == 0, " completed : %f %\n",
                 ((float)paths_head_global / (float)n_paths_) * 100.f)
  }

  RenderKernelLauncher::saveStatistics();
}

template <typename DeviceScene>
void StreamingVolPTmk<DeviceScene>::reset() {
  RenderKernelLauncher::reset();
  uint zero = 0;
  cudaMemcpyToSymbol(StreamingVolPTmk_kernel::d_paths_head_global, &zero,
                     sizeof(uint));
  seed_ += n_paths_;
  cudaMemcpyToSymbol(StreamingVolPTmk_kernel::c_seed, &seed_, sizeof(uint));
}

//---------------------------------------STREAMING-SK-----------------------------------------------

template <typename DeviceScene>
void StreamingVolPTsk<DeviceScene>::init() {
  // Get suggested block size from CUDA API
  int minGridSize;
  int blockSize;
  void* kernel = StreamingVolPTsk_kernel::d_render<
      STREAMING_THREADS_BLOCK, STREAMING_ITEMS_PER_THREAD,
      SimpleVolumeDeviceScene<DeviceMedium, GGX>>;
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel);

  COUT_DEBUG("CUDA suggested block size for StreamingVolPTsk::d_render: "
             << blockSize);
  COUT_DEBUG("CUDA suggested minimum grid size for StreamingVolPTsk::d_render: "
             << minGridSize);

  cuda_config_.block_size.x = STREAMING_THREADS_BLOCK;
  maxOccupancyGrid(cuda_config_, kernel);

  if (cuda_config_.device_properties.sharedMemPerMultiprocessor <
      STREAMING_SHARED_MEMORY * (float)cuda_config_.grid_size.x /
          (float)cuda_config_.device_properties.multiProcessorCount) {
    cuda_config_.grid_size.x =
        cuda_config_.device_properties.multiProcessorCount;
  }

  COUT_DEBUG("Cuda Config : <<< ("
             << cuda_config_.grid_size.x << "," << cuda_config_.grid_size.y
             << "," << cuda_config_.grid_size.z << ") , ("
             << cuda_config_.block_size.x << "," << cuda_config_.block_size.y
             << "," << cuda_config_.block_size.z << ") >>> \n")
  COUT_DEBUG("Occupancy d_render : " << calculateOccupancy(cuda_config_)
                                     << "% \n")
  n_threads_ = cuda_config_.grid_size.x * cuda_config_.block_size.x;
}

template <typename DeviceScene>
void StreamingVolPTsk<DeviceScene>::allocateDeviceMemory() {
  std::vector<Threads*> threads_buffers{&d_threads_};

  for (int i = 0; i < threads_buffers.size(); i++) {
    auto thread_buffer = threads_buffers[i];
    checkCudaErrors(
        cudaMalloc(&thread_buffer->paths.rays.o,
                   STREAMING_ITEMS_PER_THREAD * n_threads_ * sizeof(float3)));
    checkCudaErrors(
        cudaMalloc(&thread_buffer->paths.rays.d,
                   STREAMING_ITEMS_PER_THREAD * n_threads_ * sizeof(float3)));
    checkCudaErrors(
        cudaMalloc(&thread_buffer->paths.throughputs,
                   STREAMING_ITEMS_PER_THREAD * n_threads_ * sizeof(float4)));
    checkCudaErrors(
        cudaMalloc(&thread_buffer->image_ids,
                   STREAMING_ITEMS_PER_THREAD * n_threads_ * sizeof(uint)));
  }
}

template <typename DeviceScene>
void StreamingVolPTsk<DeviceScene>::releaseDeviceMemory() {
  std::vector<Threads*> threads_buffers{&d_threads_};

  for (int i = 0; i < threads_buffers.size(); i++) {
    auto thread_buffer = threads_buffers[i];
    checkCudaErrors(cudaFree(thread_buffer->paths.rays.o));
    checkCudaErrors(cudaFree(thread_buffer->paths.rays.d));
    checkCudaErrors(cudaFree(thread_buffer->paths.throughputs));
    checkCudaErrors(cudaFree(thread_buffer->image_ids));
  }
}

template <typename DeviceScene>
void StreamingVolPTsk<DeviceScene>::launchRender() {
  RenderKernelLauncher::loadStatistics();
  StreamingVolPTsk_kernel::d_render<STREAMING_THREADS_BLOCK,
                                    STREAMING_ITEMS_PER_THREAD>
      <<<cuda_config_.grid_size, cuda_config_.block_size,
         cuda_config_.dynamic_shared_memory>>>(d_threads_, d_output_,
                                               device_scene_);
  CHECK_CUDA_ERROR("render check ");
  RenderKernelLauncher::saveStatistics();
}

template <typename DeviceScene>
void StreamingVolPTsk<DeviceScene>::reset() {
  RenderKernelLauncher::reset();
  uint zero = 0;
  cudaMemcpyToSymbol(StreamingVolPTsk_kernel::d_paths_head_global, &zero,
                     sizeof(uint));
  seed_++;
  cudaMemcpyToSymbol(StreamingVolPTsk_kernel::c_seed, &seed_, sizeof(uint));
}

//---------------------------------------SORTING-SK-----------------------------------------------

template <class DeviceScene>
void SortingVolPTsk<DeviceScene>::init() {
  // Get suggested block size from CUDA API
  int minGridSize;
  int blockSize;
  void* kernel = SortingVolPTsk_kernel::d_render<
      STREAMING_THREADS_BLOCK, STREAMING_ITEMS_PER_THREAD,
      SimpleVolumeDeviceScene<DeviceMedium, GGX>>;
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel);

  COUT_DEBUG(
      "CUDA suggested block size for SortingVolPTsk::d_render: " << blockSize);
  COUT_DEBUG("CUDA suggested minimum grid size for SortingVolPTsk::d_render: "
             << minGridSize);

  cuda_config_.block_size.x = STREAMING_THREADS_BLOCK;
  maxOccupancyGrid(cuda_config_, kernel);

  if (cuda_config_.device_properties.sharedMemPerMultiprocessor <
      STREAMING_SHARED_MEMORY * (float)cuda_config_.grid_size.x /
          (float)cuda_config_.device_properties.multiProcessorCount) {
    cuda_config_.grid_size.x =
        cuda_config_.device_properties.multiProcessorCount;
  }

  COUT_DEBUG("Cuda Config : <<< ("
             << cuda_config_.grid_size.x << "," << cuda_config_.grid_size.y
             << "," << cuda_config_.grid_size.z << ") , ("
             << cuda_config_.block_size.x << "," << cuda_config_.block_size.y
             << "," << cuda_config_.block_size.z << ") >>> \n")
  COUT_DEBUG("Occupancy d_render : " << calculateOccupancy(cuda_config_)
                                     << "% \n")
  n_threads_ = cuda_config_.grid_size.x * cuda_config_.block_size.x;
}
template <class DeviceScene>
void SortingVolPTsk<DeviceScene>::allocateDeviceMemory() {
  std::vector<Threads*> threads_buffers{&d_threads_};

  for (int i = 0; i < threads_buffers.size(); i++) {
    auto thread_buffer = threads_buffers[i];
    checkCudaErrors(
        cudaMalloc(&thread_buffer->paths.rays.o,
                   STREAMING_ITEMS_PER_THREAD * n_threads_ * sizeof(float3)));
    checkCudaErrors(
        cudaMalloc(&thread_buffer->paths.rays.d,
                   STREAMING_ITEMS_PER_THREAD * n_threads_ * sizeof(float3)));
    checkCudaErrors(
        cudaMalloc(&thread_buffer->paths.throughputs,
                   STREAMING_ITEMS_PER_THREAD * n_threads_ * sizeof(float4)));
    checkCudaErrors(
        cudaMalloc(&thread_buffer->image_ids,
                   STREAMING_ITEMS_PER_THREAD * n_threads_ * sizeof(uint)));
  }
}
template <class DeviceScene>
void SortingVolPTsk<DeviceScene>::releaseDeviceMemory() {
  std::vector<Threads*> threads_buffers{&d_threads_};

  for (int i = 0; i < threads_buffers.size(); i++) {
    auto thread_buffer = threads_buffers[i];
    checkCudaErrors(cudaFree(thread_buffer->paths.rays.o));
    checkCudaErrors(cudaFree(thread_buffer->paths.rays.d));
    checkCudaErrors(cudaFree(thread_buffer->paths.throughputs));
    checkCudaErrors(cudaFree(thread_buffer->image_ids));
  }
}

template <class DeviceScene>
void SortingVolPTsk<DeviceScene>::launchRender() {
  RenderKernelLauncher::loadStatistics();
  SortingVolPTsk_kernel::d_render<STREAMING_THREADS_BLOCK,
                                  STREAMING_ITEMS_PER_THREAD>
      <<<cuda_config_.grid_size, cuda_config_.block_size,
         cuda_config_.dynamic_shared_memory>>>(d_threads_, d_output_,
                                               device_scene_);
  // CHECK_CUDA_ERROR("render check ");
  RenderKernelLauncher::saveStatistics();
}

template <class DeviceScene>
void SortingVolPTsk<DeviceScene>::reset() {
  RenderKernelLauncher::reset();
  uint zero = 0;
  cudaMemcpyToSymbol(SortingVolPTsk_kernel::d_paths_head_global, &zero,
                     sizeof(uint));
  seed_++;
  cudaMemcpyToSymbol(SortingVolPTsk_kernel::c_seed, &seed_, sizeof(uint));
}