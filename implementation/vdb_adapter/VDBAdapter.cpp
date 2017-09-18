#include "VDBAdapter.h"

#include <openvdb/io/File.h>

#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <iostream>
#include <stdexcept>

VDBAdapter::VDBAdapter() { openvdb::initialize(); }

VDBAdapter::~VDBAdapter() = default;

void VDBAdapter::loadVDBFile(const std::string& filepath) {
  openvdb::io::File file(filepath);
  try {
    file.open();

    // Load density grid
    if (!file.hasGrid("density")) {
      file.close();
      throw std::runtime_error("VDB file does not contain a density grid");
    }
    // Grid data is managed by shared pointers (GridBase::Ptr is a
    // std::shared_ptr)
    openvdb::GridBase::Ptr baseGrid = file.readGrid("density");
    // gridPtrCast maintains shared ownership
    density_grid_ = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);

    // Load albedo grid
    if (file.hasGrid("albedo")) {
      openvdb::GridBase::Ptr baseAlbedoGrid = file.readGrid("albedo");
      albedo_grid_ = openvdb::gridPtrCast<openvdb::Vec3SGrid>(baseAlbedoGrid);
    } else {
      throw std::runtime_error("VDB file does not contain an albedo grid");
    }

    file.close();
  } catch (const openvdb::Exception& e) {
    throw std::runtime_error("OpenVDB error: " + std::string(e.what()));
  }
}

auto VDBAdapter::getGridResolution() const
    -> std::tuple<unsigned int, unsigned int, unsigned int> {
  if (!density_grid_) {
    throw std::runtime_error("Density grid not loaded");
  }

  auto bbox = density_grid_->evalActiveVoxelBoundingBox();
  auto dim = bbox.dim();
  return std::make_tuple(dim.x(), dim.y(), dim.z());
}

std::vector<float> VDBAdapter::getDensityDataAsLinearArray(
    float inactiveValue) {
  if (!density_grid_) {
    throw std::runtime_error("Density grid not loaded");
  }

  auto bbox = density_grid_->evalActiveVoxelBoundingBox();
  auto dim = bbox.dim();

  // Initialize vector with inactive value
  std::vector<float> data(dim.x() * dim.y() * dim.z(), inactiveValue);

  // Iterate over active voxels only
  for (openvdb::FloatGrid::ValueOnCIter iter = density_grid_->cbeginValueOn();
       iter; ++iter) {
    const auto& coord = iter.getCoord();
    size_t index = (coord.x() - bbox.min().x()) +
                   (dim.x() * ((coord.y() - bbox.min().y()) +
                               dim.y() * (coord.z() - bbox.min().z())));
    data[index] = *iter;
  }

  return data;
}

std::vector<float> VDBAdapter::getAlbedoDataAsLinearArray(
    std::tuple<float, float, float> inactiveColor) {
  if (!albedo_grid_) {
    throw std::runtime_error("Albedo grid not loaded");
  }

  auto bbox = albedo_grid_->evalActiveVoxelBoundingBox();
  auto dim = bbox.dim();
  auto [r, g, b] = inactiveColor;

  // Initialize vector with inactive color
  std::vector<float> albedo_data(dim.x() * dim.y() * dim.z() * 3);
  for (size_t i = 0; i < dim.x() * dim.y() * dim.z(); ++i) {
    albedo_data[i * 3] = r;
    albedo_data[(i * 3) + 1] = g;
    albedo_data[(i * 3) + 2] = b;
  }

  // Iterate over active voxels only
  for (openvdb::Vec3SGrid::ValueOnCIter iter = albedo_grid_->cbeginValueOn();
       iter; ++iter) {
    const auto& coord = iter.getCoord();
    size_t index = ((coord.x() - bbox.min().x()) +
                    dim.x() * ((coord.y() - bbox.min().y()) +
                               dim.y() * (coord.z() - bbox.min().z()))) *
                   3;
    openvdb::Vec3s color = *iter;
    albedo_data[index] = color.x();
    albedo_data[index + 1] = color.y();
    albedo_data[index + 2] = color.z();
  }

  return albedo_data;
}

auto VDBAdapter::getVolumeAABB() const
    -> std::tuple<std::tuple<float, float, float>,
                  std::tuple<float, float, float>> {
  if (!density_grid_) {
    throw std::runtime_error("Density grid not loaded");
  }

  auto bbox = density_grid_->evalActiveVoxelBoundingBox();

  // Convert to world space using the grid's transform
  auto world_min = density_grid_->indexToWorld(bbox.min());
  auto world_max = density_grid_->indexToWorld(bbox.max());

  return {std::make_tuple(world_min.x(), world_min.y(), world_min.z()),
          std::make_tuple(world_max.x(), world_max.y(), world_max.z())};
}