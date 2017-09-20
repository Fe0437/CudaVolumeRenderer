/*
 * Scene.h
 *
 *  Created on: 30/ago/2017
 *      Author: macbook
 */

#ifndef SCENE_H_
#define SCENE_H_

#include <fstream>
#include <iostream>
#include "Defines.h"
using namespace std;

template <class VolumeType>
class Volume{
	VolumeType* vol_data_;
	glm::ivec3 volume_size_;

public:

	Volume(
			VolumeType* vol_data,
			glm::vec3 volume_size
	):
		vol_data_(vol_data),
		volume_size_(volume_size)
	{}

	Volume():vol_data_(0){}

	VolumeType* getVolumeData(){ return vol_data_;}
	glm::ivec3 getVolumeSize(){ return volume_size_;}

	cudaExtent getCudaExtent(){
		cudaExtent extent;
		extent.width = volume_size_.x;
		extent.height = volume_size_.y;
		extent.depth = volume_size_.z;
		return extent;
	}

};

class Scene{

private:
	glm::vec3 box_min_;
	glm::vec3 box_max_;
	Volume<ALBEDO_T> albedo_volume_;
	Volume<FLOAT> density_volume_;

public:
	Scene(Volume<ALBEDO_T> albedo_volume, Volume<FLOAT> density_volume, glm::vec3 box_min, glm::vec3 box_max):
		albedo_volume_(albedo_volume),
		density_volume_(density_volume),
		box_min_(box_min),
		box_max_(box_max)
		{}

	Scene(){}
	~Scene() = default;

	glm::vec3 getBoxMin() const { return box_min_;}
	glm::vec3 getBoxMax() const { return box_max_;}
	Volume<ALBEDO_T> getAlbedoVolume() const { return albedo_volume_;}
	Volume<FLOAT> getDensityVolume() const { return density_volume_;}
};


class SceneBuilder{
	public:
        virtual glm::vec3 getBoxMin() = 0;
        virtual glm::vec3 getBoxMax() = 0;
        virtual Volume<ALBEDO_T> getAlbedoVolume() = 0;
        virtual Volume<FLOAT> getDensityVolume() = 0;
};

class SceneAssembler {
	SceneBuilder* builder;

public:
	SceneAssembler():builder(0){}
	~SceneAssembler()=default ;

	void setBuilder(SceneBuilder* newBuilder)
	{
		builder = newBuilder;
	}

	Scene getScene(){
		Scene scene(
				builder->getAlbedoVolume(),
				builder->getDensityVolume(),
				builder->getBoxMin(),
				builder->getBoxMax()
				);
		return scene;
	}
};

class XMLSceneBuilder : public SceneBuilder{

public:

	glm::vec3 box_min_;
	glm::vec3 box_max_;
	glm::ivec3 volume_size_;
	Volume<ALBEDO_T> albedo_volume_;
	Volume<FLOAT> density_volume_;

	XMLSceneBuilder(string filename){
		parse(filename);
	}
	~XMLSceneBuilder() = default;

	void parse(string filename){
		auto albedo_data = (ALBEDO_T*)loadVolFile<ALBEDO_T>("../../data/cgg_logo/volume_cgglogo.vol");
		albedo_volume_ = Volume<ALBEDO_T>(albedo_data, volume_size_);
		auto density_data = (FLOAT *)loadVolFile<FLOAT>("../../data/cgg_logo/volume_cgglogo.vol");
		density_volume_ = Volume<FLOAT>(density_data, volume_size_);
	}

	 virtual glm::vec3 getBoxMin(){return box_min_;}
	 virtual glm::vec3 getBoxMax(){return box_max_;}
	 virtual Volume<ALBEDO_T> getAlbedoVolume(){return albedo_volume_;}
	 virtual Volume<FLOAT> getDensityVolume(){return density_volume_;}

private:

	void vol2Raw4f(float* volData, float4* rawData){
		  for (int z = 0; z < volume_size_.z; z++)
		    for (int y = 0; y < volume_size_.y; y++)
		      for (int x = 0; x < volume_size_.x; x++)
		        {
		    	  float* p = &volData[((z*volume_size_.y + y)*volume_size_.x + x)*3];
		    	  *rawData =make_float4(float(*p),float(*(p+1)), float(*(p+2)), (float) 1);
				  //printf("color %f %f %f at position %f %f %f \n", float(*p), float(*(p + 1)), float(*(p + 2)), float(x)/volume_size_.x, float(y) / volume_size_.y, float(z) / volume_size_.z);
		    	  rawData++;
		        }
		}

	float vol2Rawf(float* volData, float* rawData){
	  float max = 0;
	  for (int z = 0; z < volume_size_.z; z++)
	    for (int y = 0; y < volume_size_.y; y++)
	      for (int x = 0; x < volume_size_.x; x++)
	        {
	    	  *rawData = volData[((z*volume_size_.y + y)*volume_size_.x + x)];
	    	  //printf("data : %f \n", *rawData );
	    	  //printf("x : %d y : %d z : %d\n", x, y, z);
	    	  max = std::fmax( *rawData, max);
	    	  rawData++;
	        }

	  //printf("loading done \n");
	  return max;
	}

	// Load vol data from disk
	template <class VolumeType>
	void* loadVolFile(string filename, float* max = 0)
	{
	    ifstream stream (filename, ios::in | ios::binary | ios::ate);

	    if(!stream.is_open()){
	        fprintf(stderr, "Error opening file '%s'\n", (char*) filename.c_str());
	        return 0;
	    }
	    int size = (int)stream.tellg();
	    stream.seekg (0, ios::beg);

	    char header[3];
	    if (!stream.read(header, 3))
	    {
	        fprintf(stderr, "Error reading file '%s'\n", (char*) filename.c_str());
	        return 0;
	    }
	    size-=3;

	    if (header[0] != 'V' || header[1] != 'O' || header[2] != 'L')
	        fprintf(stderr, "Encountered an invalid volume data file "
	            "(incorrect header identifier)");

	    uint8_t version;
	    stream.read((char*)&version, sizeof(uint8_t));
	    size-=sizeof(uint8_t);

	    if (version != 3)
	        fprintf(stderr, "Encountered an invalid volume data file "
	            "(incorrect file version) %d", version);

	    int type;
	    if(!stream.read((char*)&type, sizeof(int))) fprintf(stderr, "Encountered an invalid volume data file "
	            "(incorrect file type) %d", type);
	    printf("type of file %d \n", type);
	    size-=sizeof(int);

	    stream.read((char*)&volume_size_.x, sizeof(int));
	    stream.read((char*)&volume_size_.y, sizeof(int));
	    stream.read((char*)&volume_size_.z, sizeof(int));
	    size-=3*sizeof(int);

	    int channels;
	    stream.read((char*)&channels, sizeof(int));
	    printf("number of channels %d \n", channels);
	    size-=sizeof(int);

	    stream.read((char*)&box_min_.x, sizeof(float));
	    stream.read((char*)&box_min_.y, sizeof(float));
	    stream.read((char*)&box_min_.z, sizeof(float));
	    stream.read((char*)&box_max_.x, sizeof(float));
	    stream.read((char*)&box_max_.y, sizeof(float));
	    stream.read((char*)&box_max_.z, sizeof(float));
	    size-=6*sizeof(float);

	    float* voldata = (float*) malloc(size);
	    if(!stream.read ((char*)voldata, size))
	    {
	    	fprintf(stderr, "Error reading data from file '%s'\n", (char*) filename.c_str());
	    }
	    stream.close();

	    size_t element_size;
	    void* rawData;

	    if(channels == 3){
	    	element_size = sizeof(float4);
	        //uchar4* rawData = (uchar4*)malloc(volume_size_.x* volume_size_.y* volume_size_.z* 4);
	        rawData = malloc(volume_size_.x* volume_size_.y* volume_size_.z* element_size);
	        vol2Raw4f(voldata, (float4*) rawData);
	    }
	    else if(channels >= 1)
	    {
	    	element_size = sizeof(float);
	    	rawData = malloc(volume_size_.x* volume_size_.y* volume_size_.z* element_size);
	    	float _max = vol2Rawf(voldata, (float*)rawData);
	    	if( max != 0 ) *max = _max;
	   	}

	    free(voldata);
	    return rawData;
	}
};



#endif /* SCENE_H_ */
