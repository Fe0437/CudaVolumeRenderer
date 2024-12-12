#pragma once


//boost xml parser gives an error when compiling with Cuda
#include <pugixml.hpp>
#include <boost/filesystem.hpp>

#include "Camera.h"
#include "Scene.h"
#include "Defines.h" 
#include "Utilities.h"
#include "Volume.h"
#include "Medium.h"

class XmlSceneBuilder : public SceneBuilder {
	typedef HostMedium Medium;

public:
	Camera* camera_;
	AbstractGeometry* geometry_;
	uint3 volume_size_;
	Medium medium_;
	float3 box_min_;
	float3 box_max_;

	XmlSceneBuilder(std::string filename) {
		auto result = parse(filename);

		if (!result) {
			LOG_DEBUG(result.description())
				throw std::invalid_argument(result.description());
		}
	}

	~XmlSceneBuilder() = default;

	pugi::xml_parse_result parse(std::string xml_filepath) {

		using namespace pugi;
		using namespace boost::filesystem;
		xml_document doc;

		path basepath(xml_filepath);
		basepath.remove_filename();

		//parsing

		pugi::xml_parse_result result = doc.load_file(xml_filepath.c_str());
		if (!result) { return result; }

		path albedo_vol_filepath;
		path density_vol_filepath;
		float medium_scale;

		auto medium = doc.child("scene").find_child_by_attribute("medium", "type", "heterogeneous");

		auto albedo = medium.find_child_by_attribute("volume", "name", "albedo");
		auto density = medium.find_child_by_attribute("volume", "name", "density");

		if (albedo.attribute("type").as_string() == std::string("gridvolume")) {

			albedo_vol_filepath = path(albedo.child("string").attribute("value").value());
		}
		else {
			pugi::xml_parse_result result;
			result.status = status_internal_error;
			return result;
		}

		if (density.attribute("type").as_string() == std::string("gridvolume")) {
			density_vol_filepath = path(density.child("string").attribute("value").value());
		}
		else {
			pugi::xml_parse_result result;
			result.status = status_internal_error;
			return result;
		}

		medium_scale = medium.find_child_by_attribute("float", "name", "scale").attribute("value").as_float();

		//loading into medium
		auto density_data = (float *)loadVolFile<float>((basepath / density_vol_filepath).string(), &medium_.max_density);
		medium_.density_volume = Volume<float>(density_data, volume_size_);

		auto albedo_data = (float4*)loadVolFile<float4>((basepath / albedo_vol_filepath).string());
		medium_.albedo_volume = Volume<float4>(albedo_data, volume_size_);

		medium_.density_AABB = AABB(box_min_, box_max_);
		medium_.scale = medium_scale;

		setupCamera(doc);

		return result;
	}

	/*setup and return the camera*/
	void setupCamera(const pugi::xml_document& doc) {
		auto camera_xml = doc.child("scene").find_child_by_attribute("sensor", "type", "perspective");
		auto fov = camera_xml.find_child_by_attribute("float", "name", "fov");
		auto film = camera_xml.find_child_by_attribute("film", "type", "hdrfilm");

		int w, h;
		if (film) {
			auto width = film.find_child_by_attribute("integer", "name", "width");
			auto height = film.find_child_by_attribute("integer", "name", "height");
			w = width.attribute("value").as_int();
			h = width.attribute("value").as_int();
		}
		else {
			w = 400;
			h = 400;
		}

		camera_ = new Camera(w, h, fov.attribute("value").as_float());
	}

	virtual AbstractGeometry* getGeometry() { return geometry_; }
	virtual Camera* getCamera() { return camera_; }
	virtual Medium getMedium() { return medium_; }

private:

	void vol2Raw4f(float* volData, float4* rawData) {
		for (uint z = 0; z < volume_size_.z; z++)
			for (uint y = 0; y < volume_size_.y; y++)
				for (uint x = 0; x < volume_size_.x; x++)
				{
					float* p = &volData[((z*volume_size_.y + y)*volume_size_.x + x) * 3];
					*rawData = make_float4(float(*p), float(*(p + 1)), float(*(p + 2)), (float)1);
					rawData++;
				}
	}

	float vol2Rawf(float* volData, float* rawData) {
		float max = 0;
		for (uint z = 0; z < volume_size_.z; z++)
			for (uint y = 0; y < volume_size_.y; y++)
				for (uint x = 0; x < volume_size_.x; x++)
				{
					*rawData = volData[((z*volume_size_.y + y)*volume_size_.x + x)];
					max = std::fmax(std::fmin(1.f, *rawData), max);
					rawData++;
				}

		return max;
	}

	// Load vol data from disk
	template <class VolumeType>
	void* loadVolFile(std::string filename, float* max = 0)
	{
		std::ifstream stream(filename, std::ios::in | std::ios::binary | std::ios::ate);

		if (!stream.is_open()) {
			fprintf(stderr, "Error opening file '%s'\n", (char*)filename.c_str());
			return 0;
		}
		int size = (int)stream.tellg();
		stream.seekg(0, std::ios::beg);

		char header[3];
		if (!stream.read(header, 3))
		{
			fprintf(stderr, "Error reading file '%s'\n", (char*)filename.c_str());
			return 0;
		}
		size -= 3;

		if (header[0] != 'V' || header[1] != 'O' || header[2] != 'L')
			fprintf(stderr, "Encountered an invalid volume data file "
				"(incorrect header identifier)");

		uint8_t version;
		stream.read((char*)&version, sizeof(uint8_t));
		size -= sizeof(uint8_t);

		if (version != 3)
			fprintf(stderr, "Encountered an invalid volume data file "
				"(incorrect file version) %d", version);

		int type;
		if (!stream.read((char*)&type, sizeof(int))) fprintf(stderr, "Encountered an invalid volume data file "
			"(incorrect file type) %d", type);
		size -= sizeof(int);

		stream.read((char*)&volume_size_.x, sizeof(int));
		stream.read((char*)&volume_size_.y, sizeof(int));
		stream.read((char*)&volume_size_.z, sizeof(int));
		size -= 3 * sizeof(int);

		COUT_DEBUG("Resolution : " << volume_size_.x << ", " << volume_size_.y << ", " << volume_size_.z << "\n")

		int channels;
		stream.read((char*)&channels, sizeof(int));
		size -= sizeof(int);

		stream.read((char*)&box_min_.x, sizeof(float));
		stream.read((char*)&box_min_.y, sizeof(float));
		stream.read((char*)&box_min_.z, sizeof(float));
		stream.read((char*)&box_max_.x, sizeof(float));
		stream.read((char*)&box_max_.y, sizeof(float));
		stream.read((char*)&box_max_.z, sizeof(float));
		size -= 6 * sizeof(float);

		float* voldata = (float*)malloc(size);
		if (!stream.read((char*)voldata, size))
		{
			fprintf(stderr, "Error reading data from file '%s'\n", (char*)filename.c_str());
		}
		stream.close();

		size_t element_size;
		void* rawData;

		if (channels == 3) {
			element_size = sizeof(float4);
			//uchar4* rawData = (uchar4*)malloc(volume_size_.x* volume_size_.y* volume_size_.z* 4);
			rawData = malloc(volume_size_.x* volume_size_.y* volume_size_.z* element_size);
			vol2Raw4f(voldata, (float4*)rawData);
		}
		else if (channels >= 1)
		{
			element_size = sizeof(float);
			rawData = malloc(volume_size_.x* volume_size_.y* volume_size_.z* element_size);
			float _max = vol2Rawf(voldata, (float*)rawData);
			if (max != 0) *max = _max;
		}

		free(voldata);
		return rawData;
	}
};
