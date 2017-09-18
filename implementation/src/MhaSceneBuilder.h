#pragma once

#include "Defines.h"
#ifdef MHA_SUPPORT

#include <vtkImageData.h>
#include <vtkImageResample.h>
#include <vtkMetaImageReader.h>

#include <boost/filesystem.hpp>

#include "Medium.h"
#include "Utilities.h"
#include "Volume.h"

class MhaSceneBuilder : public SceneBuilder {
  typedef HostMedium Medium;

 public:
  Camera* camera_;
  AbstractGeometry* geometry_;
  uint3 volume_size_;
  Medium medium_;
  float3 box_min_;
  float3 box_max_;
  float2 smooth_step_edges_;

  MhaSceneBuilder(std::string filename) { parse(filename); }

  ~MhaSceneBuilder() = default;

  void parse(std::string mha_filepath) {
    smooth_step_edges_.x = 0.2;
    smooth_step_edges_.y = 0.6;

    // loading into medium
    auto density_data =
        (float*)loadMhdFile<float>(mha_filepath, &medium_.max_density);
    medium_.density_volume = Volume<float>(density_data, volume_size_);

    auto albedo_data = getAlbedoFromDensity(density_data, volume_size_);
    medium_.albedo_volume = Volume<float4>(albedo_data, volume_size_);

    box_min_ = make_float3(-0.5, -0.5, -0.5);
    box_max_ = make_float3(0.5, 0.5, 0.5);

    medium_.density_AABB = AABB(box_min_, box_max_);
    medium_.scale = 100.f;

    setupCamera();
  }

  /*setup and return the camera*/
  void setupCamera() { camera_ = new Camera(); }

  virtual AbstractGeometry* getGeometry() { return geometry_; }
  virtual Camera* getCamera() { return camera_; }
  virtual Medium getMedium() { return medium_; }

 private:
  float4* getAlbedoFromDensity(float* density, uint3 volume_size) {
    float func_length = 100.f;
    std::vector<float4> transferFunc;
    float start_r = 0.02;
    float start_g = 0.02;
    float start_b = 0.02;

    float end_r = 1.f;
    float end_g = 0.02;
    float end_b = 0.02;

    for (int i = 0; i < func_length * 1.f / 5.f; i++) {
      auto color =
          make_float4(start_r + i * (end_r - start_r) / func_length,
                      start_g + i * (end_g - start_g) / func_length,
                      start_b + i * (end_b - start_b) / func_length, 1.f);
      transferFunc.push_back(color);
    }

    start_r = end_r;
    start_g = end_g;
    start_b = end_b;

    end_r = 0.0f;
    end_g = 0.02;
    end_b = 1.0;

    for (int i = 0; i < func_length * 4.f / 5.f; i++) {
      auto color =
          make_float4(start_r + i * (end_r - start_r) / func_length,
                      start_g + i * (end_g - start_g) / func_length,
                      start_b + i * (end_b - start_b) / func_length, 1.f);
      transferFunc.push_back(color);
    }

    float4* rawData = (float4*)malloc(volume_size_.x * volume_size_.y *
                                      volume_size_.z * sizeof(float4));

    float* density_iter = density;
    float4* albedo_iter = rawData;

    for (uint z = 0; z < volume_size_.z; z++)
      for (uint y = 0; y < volume_size_.y; y++)
        for (uint x = 0; x < volume_size_.x; x++) {
          float v = (*density_iter) * (transferFunc.size() - 1);
          *albedo_iter = transferFunc[std::ceil(v)];
          density_iter++;
          albedo_iter++;
        }

    return rawData;
  }

  // Load vol data from disk
  template <class VolumeType>
  void* loadMhdFile(std::string mha_filepath, float* max = 0) {
    vtkImageData* input = 0;

    vtkMetaImageReader* metaReader = vtkMetaImageReader::New();
    metaReader->SetFileName(mha_filepath.c_str());
    metaReader->Update();

    // Verify that we actually have a volume
    int dim[3];
    metaReader->GetOutput()->GetDimensions(dim);
    if (dim[0] < 2 || dim[1] < 2 || dim[2] < 2) {
      cout << "Error loading data!" << endl;
    }

    vtkImageResample* resample = vtkImageResample::New();
    resample->SetInputConnection(metaReader->GetOutputPort());

    // auto spacing = metaReader->GetDataSpacing();
    // auto normalized_spacing = std::min(spacing[0], std::min(spacing[1],
    // spacing[2]));
    float normalized_spacing = 1.f;
    resample->SetOutputSpacing(normalized_spacing, normalized_spacing,
                               normalized_spacing);
    resample->SetAxisMagnificationFactor(0, 0.5);
    resample->SetAxisMagnificationFactor(1, 0.5);
    resample->SetAxisMagnificationFactor(2, 0.5);

    resample->Update();
    input = resample->GetOutput();
    input->GetDimensions(dim);

    volume_size_.x = (uint)dim[0];
    volume_size_.y = (uint)dim[1];
    volume_size_.z = (uint)dim[2];

    LOG_DEBUG("VOLUME SIZE %d %d %d \n", volume_size_.x, volume_size_.y,
              volume_size_.z)

    size_t element_size = sizeof(float);
    float* rawData = (float*)malloc(volume_size_.x * volume_size_.y *
                                    volume_size_.z * element_size);

    float* iter = rawData;

    float _max = input->GetScalarTypeMin();
    float _min = input->GetScalarTypeMax();

    for (uint z = 0; z < volume_size_.z; z++)
      for (uint y = 0; y < volume_size_.y; y++)
        for (uint x = 0; x < volume_size_.x; x++) {
          *iter = input->GetScalarComponentAsFloat(x, y, z, 0);

          _max = std::fmax(*iter, _max);
          _min = std::fmin(*iter, _min);

          iter++;
        }

    // normalizing

    iter = rawData;
    for (uint z = 0; z < volume_size_.z; z++)
      for (uint y = 0; y < volume_size_.y; y++)
        for (uint x = 0; x < volume_size_.x; x++) {
          *iter = smoothStep(smooth_step_edges_.x, smooth_step_edges_.y,
                             (*iter - _min) / (_max - _min));
          iter++;
        }

    // float _max = mhd2Rawf(input, (float*)rawData);
    if (max != 0) *max = 1.f;

    metaReader->Delete();
    resample->Delete();
    return rawData;
  }
};

#endif