#pragma once

#include <argparse/argparse.hpp>

#include "Camera.h"
#include "Scene.h"
#include "Defines.h" 
#include "Utilities.h"
#include "Volume.h"
#include "Medium.h"

class RawSceneBuilder : public SceneBuilder {
	typedef HostMedium Medium;

public:
	Camera* camera_;
	AbstractGeometry* geometry_;
	uint3 volume_size_;
	Medium medium_;
	float3 box_min_;
	float3 box_max_;

	RawSceneBuilder(std::string filename) {
		auto result = parse(filename);
	}

	~RawSceneBuilder() = default;

	bool parse(std::string raw_filepath) {
		
		volume_size_ = make_uint3(32, 32, 32);
		typedef unsigned char VolumeType;
		size_t size = volume_size_.x*volume_size_.y*volume_size_.z * sizeof(VolumeType);
		auto raw_data = (VolumeType *)loadRawFile(raw_filepath.c_str(), size);

		float* density_data = (float*)malloc(volume_size_.x*volume_size_.y*volume_size_.z * sizeof(float));
		float* p_iter = density_data;
		
		float max = 0;
		for (uint z = 0; z < volume_size_.z; z++)
			for (uint y = 0; y < volume_size_.y; y++)
				for (uint x = 0; x < volume_size_.x; x++)
				{
					*p_iter = *raw_data;
					//printf(" data :%d \n", *raw_data);
					max = std::fmax(*p_iter, max);
					p_iter++;
					raw_data++;
				}

		p_iter = density_data;

		for (uint z = 0; z < volume_size_.z; z++)
			for (uint y = 0; y < volume_size_.y; y++)
				for (uint x = 0; x < volume_size_.x; x++)
				{
					*p_iter = (*p_iter)/max;
					p_iter++;
				}

		medium_.density_volume = Volume<float>(density_data, volume_size_);
		auto albedo_data = getAlbedoFromDensity(density_data, volume_size_);

		medium_.max_density = 1;
		medium_.albedo_volume = Volume<float4>(albedo_data, volume_size_);

		box_min_ = make_float3(-0.5, -0.5, -0.5);
		box_max_ = make_float3(0.5, 0.5, 0.5);

		medium_.density_AABB = AABB(box_min_, box_max_);
		medium_.scale = 40;

		setupCamera();

		return true;
	}

	/*setup and return the camera*/
	void setupCamera() {
		camera_ = new Camera();
	}

	virtual AbstractGeometry* getGeometry() { return geometry_; }
	virtual Camera* getCamera() { return camera_; }
	virtual Medium getMedium() { return medium_; }

private:

	float4* getAlbedoFromDensity(float* density, uint3 volume_size) {
		float func_length = 100.f;
		std::vector<float4> transferFunc;
		float start_r = 0.02;
		float start_g = 0.2;
		float start_b = 0.02;

		float end_r = 1.f;
		float end_g = 0.02;
		float end_b = 0.02;

		for (int i = 0; i < func_length * 1.f / 5.f; i++) {
			auto color = make_float4(
				start_r + i*(end_r - start_r) / func_length,
				start_g + i*(end_g - start_g) / func_length,
				start_b + i*(end_b - start_b) / func_length,
				1.f
			);
			transferFunc.push_back(color);
		}

		start_r = end_r;
		start_g = end_g;
		start_b = end_b;

		end_r = 0.0f;
		end_g = 0.02;
		end_b = 1.0;

		for (int i = 0; i < func_length * 4.f / 5.f; i++) {
			auto color = make_float4(
				start_r + i*(end_r - start_r) / func_length,
				start_g + i*(end_g - start_g) / func_length,
				start_b + i*(end_b - start_b) / func_length,
				1.f
			);
			transferFunc.push_back(color);
		}

		float4* rawData = (float4*)malloc(volume_size_.x* volume_size_.y* volume_size_.z * sizeof(float4));

		float* density_iter = density;
		float4* albedo_iter = rawData;

		for (uint z = 0; z < volume_size_.z; z++)
			for (uint y = 0; y < volume_size_.y; y++)
				for (uint x = 0; x < volume_size_.x; x++)
				{
					float v = (*density_iter)*(transferFunc.size() - 1);
					*albedo_iter = transferFunc[std::ceil(v)];
					density_iter++;
					albedo_iter++;
				}

		return rawData;
	}

	// Load raw data from disk
	void *loadRawFile(const char *filename, size_t size)
	{
		FILE *fp = fopen(filename, "rb");

		if (!fp)
		{
			fprintf(stderr, "Error opening file '%s'\n", filename);
			return 0;
		}

		void *data = malloc(size);
		size_t read = fread(data, 1, size, fp);
		fclose(fp);

#if defined(_MSC_VER_)
		printf("Read '%s', %Iu bytes\n", filename, read);
#else
		printf("Read '%s', %zu bytes\n", filename, read);
#endif

		return data;
	}

};
