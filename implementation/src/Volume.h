#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

#include "CVRMath.h"
#include "Geometry.h"
#include "HG.h"
#include "Rng.h"
#include "Utilities.h"

struct PhaseFunction {
  __host__ __device__ virtual float3 sample(const float3& dir,
                                            Rng& rng) const = 0;
};

struct HG {
  float g = 0.f;

  __host__ __device__ HG() {}
  __host__ __device__ HG(const HG& other) : g(other.g) {}
  __host__ __device__ __forceinline__ float3 sample(const float3& dir,
                                                    Rng& rng) const {
    float2 rnd = rng.getFloat2();
    return ImportanceSampleHG(dir, g, rnd.x, rnd.y);
  }
};

template <class VolumeType>
struct DeviceVolume {
  uint3 grid_resolution{};
  cudaTextureObject_t volume_tex{};

  __host__ __device__ DeviceVolume() {}
  __host__ __device__ DeviceVolume(const DeviceVolume& other)
      : grid_resolution(other.grid_resolution), volume_tex(other.volume_tex) {}

  __device__ __forceinline__ float3 volumeToGrid(float3 p) {
    p.x = p.x * (grid_resolution.x - 1);
    p.y = p.y * (grid_resolution.y - 1);
    p.z = p.z * (grid_resolution.z - 1);
    return p;
  }

  __device__ __forceinline__ VolumeType operator()(float3 p) {
    float3 coord = volumeToGrid(p);

#ifdef MITSUBA_COMPARABLE
    const int x1 = floorf(coord.x), y1 = floorf(coord.y), z1 = floorf(coord.z),
              x2 = x1 + 1, y2 = y1 + 1, z2 = z1 + 1;

    const float fx = coord.x - x1, fy = coord.y - y1, fz = coord.z - z1,
                _fx = 1.0f - fx, _fy = 1.0f - fy, _fz = 1.0f - fz;

    const VolumeType d000 = get(x1, y1, z1), d001 = get(x2, y1, z1),
                     d010 = get(x1, y2, z1), d011 = get(x2, y2, z1),
                     d100 = get(x1, y1, z2), d101 = get(x2, y1, z2),
                     d110 = get(x1, y2, z2), d111 = get(x2, y2, z2);

    return ((d000 * _fx + d001 * fx) * _fy + (d010 * _fx + d011 * fx) * fy) *
               _fz +
           ((d100 * _fx + d101 * fx) * _fy + (d110 * _fx + d111 * fx) * fy) *
               fz;
#else
    return get(int(coord.x), int(coord.y), int(coord.z));
#endif
  }

  __device__ __forceinline__ VolumeType get(uint x, uint y, uint z);
  __device__ __forceinline__ VolumeType get(uint x);
};

template <class VolumeType>
struct HostDeviceVolume {
  uint3 grid_resolution{};
  cudaTextureObject_t volume_tex{};

  __device__ __forceinline__ float3 volumeToGrid(float3 p) {
    p.x = p.x * (grid_resolution.x - 1);
    p.y = p.y * (grid_resolution.y - 1);
    p.z = p.z * (grid_resolution.z - 1);
    return p;
  }

  __device__ __forceinline__ VolumeType operator()(float3 p) {
    float3 coord = volumeToGrid(p);

#ifdef MITSUBA_COMPARABLE

    const int x1 = floorf(coord.x), y1 = floorf(coord.y), z1 = floorf(coord.z),
              x2 = x1 + 1, y2 = y1 + 1, z2 = z1 + 1;

    const float fx = coord.x - x1, fy = coord.y - y1, fz = coord.z - z1,
                _fx = 1.0f - fx, _fy = 1.0f - fy, _fz = 1.0f - fz;

    const VolumeType d000 = get(x1, y1, z1), d001 = get(x2, y1, z1),
                     d010 = get(x1, y2, z1), d011 = get(x2, y2, z1),
                     d100 = get(x1, y1, z2), d101 = get(x2, y1, z2),
                     d110 = get(x1, y2, z2), d111 = get(x2, y2, z2);

    return ((d000 * _fx + d001 * fx) * _fy + (d010 * _fx + d011 * fx) * fy) *
               _fz +
           ((d100 * _fx + d101 * fx) * _fy + (d110 * _fx + d111 * fx) * fy) *
               fz;
#else
    return get(int(coord.x), int(coord.y), int(coord.z));
#endif
  }

  __device__ __forceinline__ VolumeType get(uint x, uint y, uint z);
  __device__ __forceinline__ VolumeType get(uint x);
};

template <class VOLUME_TYPE>
class Volume {
 public:
  using VolumeType = VOLUME_TYPE;

 private:
  std::vector<VolumeType> vol_data_{};

 public:
  uint3 grid_resolution{};

  __host__ __device__ Volume(std::vector<VolumeType>&& vol_data,
                             uint3 _grid_resolution)
      : vol_data_(std::move(vol_data)), grid_resolution(_grid_resolution) {
    if (vol_data_.size() !=
        grid_resolution.x * grid_resolution.y * grid_resolution.z) {
      throw std::runtime_error(
          "Volume data size does not match grid resolution");
    }
  }

  __host__ __device__ Volume() = default;
  __host__ __device__ Volume(const Volume& other) = default;
  __host__ __device__ Volume& operator=(const Volume& other) = default;
  __host__ __device__ Volume(Volume&& other) = default;
  __host__ __device__ Volume& operator=(Volume&& other) = default;

  auto getVolumeData() -> VolumeType* { return vol_data_.data(); }
  auto getVolumeData() const -> const VolumeType* { return vol_data_.data(); }
  [[nodiscard]] uint3 getVolumeSize() const { return grid_resolution; }

  auto operator()(float3 p) const -> VolumeType {
    return vol_data_[int(p.x) + grid_resolution.x * int(p.y) +
                     grid_resolution.x * grid_resolution.y * int(p.z)];
  }

  [[nodiscard]] auto getBytes() const -> size_t {
    return vol_data_.size() * sizeof(VolumeType);
  }

  [[nodiscard]] cudaExtent getCudaExtent() const {
    cudaExtent extent;
    extent.width = grid_resolution.x;
    extent.height = grid_resolution.y;
    extent.depth = grid_resolution.z;
    return extent;
  }

  void ZYXToMortonOrder() {
    std::vector<VolumeType> temp(vol_data_.size());
    for (int i = 0; i < grid_resolution.x; i++) {
      for (int j = 0; j < grid_resolution.y; j++) {
        for (int k = 0; k < grid_resolution.z; k++) {
          int id = getMortonIndex(i, grid_resolution.x, j, grid_resolution.y, k,
                                  grid_resolution.z);
          size_t linear_idx =
              i + grid_resolution.x * (j + grid_resolution.y * k);
          temp[id] = vol_data_[linear_idx];
        }
      }
    }
    vol_data_ = std::move(temp);
  }
};
