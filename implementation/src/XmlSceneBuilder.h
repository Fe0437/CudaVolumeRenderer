#pragma once

// boost xml parser gives an error when compiling with Cuda
#include <filesystem>
#include <memory>
#include <optional>
#include <pugixml.hpp>
#include <vector>

#include "Camera.h"
#include "Defines.h"
#include "Medium.h"
#include "Scene.h"
#include "Utilities.h"
#include "Volume.h"

class XmlSceneBuilder : public SceneBuilder {
  typedef HostMedium Medium;

 public:
  std::shared_ptr<Camera> camera_;
  std::shared_ptr<AbstractGeometry> geometry_;
  uint3 volume_size_{};
  Medium medium_{};
  float3 box_min_{};
  float3 box_max_{};

  explicit XmlSceneBuilder(const std::string& filename) {
    auto result = parse(filename);

    if (!result) {
      LOG_DEBUG(result.description())
      throw std::invalid_argument(result.description());
    }
  }

  ~XmlSceneBuilder() = default;

  pugi::xml_parse_result parse(const std::string& xml_filepath) {
    using namespace pugi;
    using namespace std::filesystem;
    xml_document doc;

    std::filesystem::path basepath(xml_filepath);
    basepath.remove_filename();

    // parsing

    pugi::xml_parse_result result = doc.load_file(xml_filepath.c_str());
    if (!result) {
      return result;
    }

    std::optional<std::filesystem::path> albedo_vol_filepath;
    std::optional<std::filesystem::path> density_vol_filepath;
    std::optional<float> medium_scale;

    auto medium = doc.child("scene").find_child_by_attribute("medium", "type",
                                                             "heterogeneous");

    auto albedo = medium.find_child_by_attribute("volume", "name", "albedo");
    auto density = medium.find_child_by_attribute("volume", "name", "density");

    if (albedo &&
        albedo.attribute("type").as_string() == std::string("gridvolume")) {
      albedo_vol_filepath = std::filesystem::path(
          albedo.child("string").attribute("value").value());
    } else {
      pugi::xml_parse_result result;
      result.status = status_internal_error;
      return result;
    }

    if (density &&
        density.attribute("type").as_string() == std::string("gridvolume")) {
      density_vol_filepath = std::filesystem::path(
          density.child("string").attribute("value").value());
    } else {
      pugi::xml_parse_result result;
      result.status = status_internal_error;
      return result;
    }

    auto scale_node = medium.find_child_by_attribute("float", "name", "scale");
    if (scale_node) {
      medium_scale = scale_node.attribute("value").as_float();
    } else {
      pugi::xml_parse_result result;
      result.status = status_internal_error;
      return result;
    }

    if (!albedo_vol_filepath || !density_vol_filepath || !medium_scale) {
      pugi::xml_parse_result result;
      result.status = status_internal_error;
      return result;
    }

    // loading into medium
    auto [density_data, max_density] =
        loadVolFile<float>((basepath / *density_vol_filepath).string());
    if (max_density) {
      medium_.max_density = *max_density;
    }
    medium_.density_volume =
        Volume<float>(std::move(density_data), volume_size_);

    auto [albedo_data, _] =
        loadVolFile<float4>((basepath / *albedo_vol_filepath).string());
    medium_.albedo_volume =
        Volume<float4>(std::move(albedo_data), volume_size_);

    medium_.density_AABB = AABB(box_min_, box_max_);
    medium_.scale = *medium_scale;

    setupCamera(doc);

    return result;
  }

  /*setup and return the camera*/
  void setupCamera(const pugi::xml_document& doc) {
    auto camera_xml = doc.child("scene").find_child_by_attribute(
        "sensor", "type", "perspective");
    auto fov = camera_xml.find_child_by_attribute("float", "name", "fov");
    auto film = camera_xml.find_child_by_attribute("film", "type", "hdrfilm");

    std::optional<int> width;
    std::optional<int> height;
    std::optional<float> fov_value;

    if (film) {
      auto width_node =
          film.find_child_by_attribute("integer", "name", "width");
      auto height_node =
          film.find_child_by_attribute("integer", "name", "height");
      if (width_node && height_node) {
        width = width_node.attribute("value").as_int();
        height = height_node.attribute("value").as_int();
      }
    }

    if (fov) {
      fov_value = fov.attribute("value").as_float();
    }

    int w = width.value_or(400);
    int h = height.value_or(400);
    float fov_val = fov_value.value_or(45.0F);

    camera_ = std::make_shared<Camera>(w, h, fov_val);
  }

  auto getGeometry() -> std::shared_ptr<AbstractGeometry> override {
    return geometry_;
  }
  auto getCamera() -> std::shared_ptr<Camera> override { return camera_; }
  auto getMedium() -> Medium override { return medium_; }

 private:
  std::vector<float4> vol2Raw4f(const std::vector<float>& volData) const {
    std::vector<float4> rawData(volume_size_.x * volume_size_.y *
                                volume_size_.z);
    for (uint z = 0; z < volume_size_.z; z++) {
      for (uint y = 0; y < volume_size_.y; y++) {
        for (uint x = 0; x < volume_size_.x; x++) {
          size_t vol_idx = ((z * volume_size_.y + y) * volume_size_.x + x) * 3;
          size_t raw_idx = (z * volume_size_.y + y) * volume_size_.x + x;
          rawData[raw_idx] = make_float4(volData[vol_idx], volData[vol_idx + 1],
                                         volData[vol_idx + 2], 1.0F);
        }
      }
    }
    return rawData;
  }

  std::pair<std::vector<float>, float> vol2Rawf(
      const std::vector<float>& volData) const {
    std::vector<float> rawData(volume_size_.x * volume_size_.y *
                               volume_size_.z);
    float max = 0;
    for (uint z = 0; z < volume_size_.z; z++) {
      for (uint y = 0; y < volume_size_.y; y++) {
        for (uint x = 0; x < volume_size_.x; x++) {
          size_t idx = (z * volume_size_.y + y) * volume_size_.x + x;
          rawData[idx] = volData[idx];
          max = std::max(std::min(1.0F, rawData[idx]), max);
        }
      }
    }
    return {rawData, max};
  }

  // Load vol data from disk
  template <class VolumeType>
  std::pair<std::vector<VolumeType>, std::optional<float>> loadVolFile(
      const std::string& filename) {
    std::ifstream stream(filename, std::ios::binary | std::ios::ate);
    if (!stream) {
      throw std::runtime_error("Error opening file '" + filename + "'");
    }

    auto size = stream.tellg();
    stream.seekg(0, std::ios::beg);

    char header[3];
    if (!stream.read(header, 3)) {
      throw std::runtime_error("Error reading file '" + filename + "'");
    }
    size -= 3;

    if (header[0] != 'V' || header[1] != 'O' || header[2] != 'L') {
      throw std::runtime_error(
          "Invalid volume data file (incorrect header identifier)");
    }

    uint8_t version;
    stream.read(reinterpret_cast<char*>(&version), sizeof(uint8_t));
    size -= sizeof(uint8_t);

    if (version != 3) {
      throw std::runtime_error(
          "Invalid volume data file (incorrect file version)");
    }

    int type;
    stream.read(reinterpret_cast<char*>(&type), sizeof(int));
    size -= sizeof(int);

    stream.read(reinterpret_cast<char*>(&volume_size_.x), sizeof(int));
    stream.read(reinterpret_cast<char*>(&volume_size_.y), sizeof(int));
    stream.read(reinterpret_cast<char*>(&volume_size_.z), sizeof(int));
    size -= 3 * sizeof(int);

    COUT_DEBUG("Resolution : " << volume_size_.x << ", " << volume_size_.y
                               << ", " << volume_size_.z << "\n")

    int channels;
    stream.read(reinterpret_cast<char*>(&channels), sizeof(int));
    size -= sizeof(int);

    stream.read(reinterpret_cast<char*>(&box_min_.x), sizeof(float));
    stream.read(reinterpret_cast<char*>(&box_min_.y), sizeof(float));
    stream.read(reinterpret_cast<char*>(&box_min_.z), sizeof(float));
    stream.read(reinterpret_cast<char*>(&box_max_.x), sizeof(float));
    stream.read(reinterpret_cast<char*>(&box_max_.y), sizeof(float));
    stream.read(reinterpret_cast<char*>(&box_max_.z), sizeof(float));
    size -= 6 * sizeof(float);

    std::vector<float> voldata(size / sizeof(float));
    if (!stream.read(reinterpret_cast<char*>(voldata.data()), size)) {
      throw std::runtime_error("Error reading data from file '" + filename +
                               "'");
    }

    if constexpr (std::is_same_v<VolumeType, float4>) {
      assert(channels == 3);
      return {vol2Raw4f(voldata), std::nullopt};
    } else if constexpr (std::is_same_v<VolumeType, float>) {
      assert(channels == 1);
      auto [rawData, max] = vol2Rawf(voldata);
      return {std::move(rawData), std::make_optional(max)};
    } else {
      throw std::runtime_error("Unsupported volume type");
    }
  }
};
