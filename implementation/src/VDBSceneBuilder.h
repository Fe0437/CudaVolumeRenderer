#pragma once

// 1. CUDA headers first
#include <cuda_runtime.h>

// 2. Third-party libraries
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/Interpolation.h>

// 3. System/Standard headers
#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

// 4. Project headers
#include "Camera.h"
#include "Defines.h"
#include "Geometry.h"
#include "Medium.h"
#include "Scene.h"
#include "Utilities.h"
#include "VDBAdapter.h"
#include "Volume.h"

class VDBSceneBuilder : public SceneBuilder {
  typedef HostMedium Medium;

 public:
  std::shared_ptr<Camera> camera_;
  std::shared_ptr<AbstractGeometry> geometry_;
  uint3 volume_size_{};
  Medium medium_{};
  float3 box_min_{};
  float3 box_max_{};
  float2 smooth_step_edges_{};

  explicit VDBSceneBuilder(std::string filename) {
    VDBAdapter vdbAdapter;
    vdbAdapter.loadVDBFile(filename);

    // Get the grid's natural resolution
    auto [dim_x, dim_y, dim_z] = vdbAdapter.getGridResolution();
    volume_size_ = {dim_x, dim_y, dim_z};

    // Get density and albedo data as vectors
    auto density_vec = vdbAdapter.getDensityDataAsLinearArray(0.0F);
    auto albedo_data =
        vdbAdapter.getAlbedoDataAsLinearArray({0.0F, 0.0F, 0.0F});

    // Calculate max density
    medium_.max_density =
        *std::max_element(density_vec.begin(), density_vec.end());

    medium_.density_volume =
        Volume<float>(std::move(density_vec), volume_size_);

    // Convert albedo data to float4 vector
    std::vector<float4> albedo_vec(volume_size_.x * volume_size_.y *
                                   volume_size_.z);
    for (size_t i = 0; i < albedo_vec.size(); ++i) {
      albedo_vec[i] = make_float4(albedo_data[i * 3], albedo_data[(i * 3) + 1],
                                  albedo_data[(i * 3) + 2], 1.0F);
    }
    medium_.albedo_volume = Volume<float4>(std::move(albedo_vec), volume_size_);

    // Get world-space AABB
    auto [min_tuple, max_tuple] = vdbAdapter.getVolumeAABB();
    auto [min_x, min_y, min_z] = min_tuple;
    auto [max_x, max_y, max_z] = max_tuple;

    box_min_ = make_float3(-0.5, -0.5, -0.5);
    box_max_ = make_float3(0.5, 0.5, 0.5);
    medium_.density_AABB = AABB(box_min_, box_max_);
    medium_.scale = 100.F;

    setupCamera();
  }

  ~VDBSceneBuilder() = default;

  void setupCamera() { camera_ = std::make_shared<Camera>(); }

  auto getGeometry() -> std::shared_ptr<AbstractGeometry> override {
    return geometry_;
  }
  auto getCamera() -> std::shared_ptr<Camera> override { return camera_; }
  auto getMedium() -> Medium override { return medium_; }
};
