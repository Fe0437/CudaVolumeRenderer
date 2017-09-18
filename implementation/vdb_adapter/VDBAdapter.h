#pragma once

#include <openvdb/openvdb.h>

#include <string>
#include <tuple>
#include <vector>

class VDBAdapter {
 public:
  VDBAdapter();
  ~VDBAdapter();

  void loadVDBFile(const std::string& filepath);

  // Get the grid's resolution (number of voxels in each dimension)
  auto getGridResolution() const
      -> std::tuple<unsigned int, unsigned int, unsigned int>;

  // Get data arrays using the grid's natural resolution
  std::vector<float> getDensityDataAsLinearArray(float inactiveValue);
  std::vector<float> getAlbedoDataAsLinearArray(
      std::tuple<float, float, float> inactiveColor);

  // Get the volume bounds in world coordinates
  auto getVolumeAABB() const -> std::tuple<std::tuple<float, float, float>,
                                           std::tuple<float, float, float>>;

 private:
  openvdb::FloatGrid::Ptr density_grid_{};
  openvdb::Vec3SGrid::Ptr albedo_grid_{};
};