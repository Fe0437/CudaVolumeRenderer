#pragma once

// 1. CUDA headers first
#include <cuda_runtime.h>

#include <fstream>
#include <memory>
#include <span>  // C++20
#include <string>
#include <vector>

#include "Camera.h"
#include "Geometry.h"
#include "Medium.h"
#include "Scene.h"
#include "Volume.h"

class RawSceneBuilder : public SceneBuilder {
  typedef HostMedium Medium;

 public:
  std::shared_ptr<Camera> camera_;
  std::shared_ptr<AbstractGeometry> geometry_;
  uint3 volume_size_{};
  Medium medium_{};
  float3 box_min_{};
  float3 box_max_{};

  explicit RawSceneBuilder(std::string filename) {
    auto result = parse(filename);
  }

  ~RawSceneBuilder() = default;

  auto parse(std::string raw_filepath) -> bool {
    volume_size_ = make_uint3(32, 32, 32);
    using VolumeType = unsigned char;
    size_t size =
        volume_size_.x * volume_size_.y * volume_size_.z * sizeof(VolumeType);

    // Use vector for raw data
    std::vector<unsigned char> raw_data(size);
    if (!loadRawFile(raw_filepath.c_str(), std::span(raw_data))) {
      return false;
    }

    std::vector<float> density_data(volume_size_.x * volume_size_.y *
                                    volume_size_.z);

    float max = 0;
    for (uint z = 0; z < volume_size_.z; z++) {
      for (uint y = 0; y < volume_size_.y; y++) {
        for (uint x = 0; x < volume_size_.x; x++) {
          size_t idx =
              x + y * volume_size_.x + z * volume_size_.x * volume_size_.y;
          density_data[idx] = raw_data[idx];
          max = std::fmax(density_data[idx], max);
        }
      }
    }

    for (auto& val : density_data) {
      val /= max;
    }

    auto albedo_data = getAlbedoFromDensity(density_data, volume_size_);
    medium_.density_volume =
        Volume<float>(std::move(density_data), volume_size_);

    medium_.max_density = 1;
    medium_.albedo_volume =
        Volume<float4>(std::move(albedo_data), volume_size_);

    box_min_ = make_float3(-0.5, -0.5, -0.5);
    box_max_ = make_float3(0.5, 0.5, 0.5);

    medium_.density_AABB = AABB(box_min_, box_max_);
    medium_.scale = 40;

    setupCamera();

    return true;
  }

  /*setup and return the camera*/
  void setupCamera() { camera_ = std::make_shared<Camera>(); }

  auto getGeometry() -> std::shared_ptr<AbstractGeometry> override {
    return geometry_;
  }
  auto getCamera() -> std::shared_ptr<Camera> override { return camera_; }
  auto getMedium() -> Medium override { return medium_; }

 private:
  std::vector<float4> getAlbedoFromDensity(const std::vector<float>& density,
                                           uint3 volume_size) {
    float func_length = 100.F;
    std::vector<float4> transferFunc;
    float start_r = 0.02;
    float start_g = 0.2;
    float start_b = 0.02;

    float end_r = 1.F;
    float end_g = 0.02;
    float end_b = 0.02;

    for (int i = 0; i < func_length * 1.F / 5.F; i++) {
      auto color =
          make_float4(start_r + (i * (end_r - start_r) / func_length),
                      start_g + (i * (end_g - start_g) / func_length),
                      start_b + (i * (end_b - start_b) / func_length), 1.F);
      transferFunc.push_back(color);
    }

    start_r = end_r;
    start_g = end_g;
    start_b = end_b;

    end_r = 0.0F;
    end_g = 0.02;
    end_b = 1.0;

    for (int i = 0; i < func_length * 4.F / 5.F; i++) {
      auto color =
          make_float4(start_r + (i * (end_r - start_r) / func_length),
                      start_g + (i * (end_g - start_g) / func_length),
                      start_b + (i * (end_b - start_b) / func_length), 1.F);
      transferFunc.push_back(color);
    }

    std::vector<float4> albedo_data(volume_size_.x * volume_size_.y *
                                    volume_size_.z);

    for (size_t i = 0; i < density.size(); i++) {
      float v = density[i] * (transferFunc.size() - 1);
      albedo_data[i] = transferFunc[std::ceil(v)];
    }

    return albedo_data;
  }

  // Load raw data from disk into provided span
  static auto loadRawFile(const char* filename, std::span<unsigned char> data)
      -> bool {
    FILE* fp = fopen(filename, "rb");
    if (fp == nullptr) {
      fprintf(stderr, "Error opening file '%s'\n", filename);
      return false;
    }

    size_t read = fread(data.data(), 1, data.size(), fp);
    fclose(fp);

#if defined(_MSC_VER_)
    printf("Read '%s', %Iu bytes\n", filename, read);
#else
    printf("Read '%s', %zu bytes\n", filename, read);
#endif
    return read == data.size();
  }
};
