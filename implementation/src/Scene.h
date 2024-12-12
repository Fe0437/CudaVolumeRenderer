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
#include "Camera.h"
#include "Geometry.h"
#include "Medium.h"

class Scene{
	typedef HostMedium Medium;
private:
	AbstractGeometry* geometry_;
	std::shared_ptr<Camera> camera_;
	Medium medium_;

public:
	Scene(Camera* camera,AbstractGeometry* geometry, Medium medium):
		camera_(camera),
		geometry_(geometry),
		medium_(medium)
		{}

	Scene(){}
	~Scene() = default;

	void growBBox(float3 &box_min, float3 &box_max) {
		geometry_->growBBox(box_min, box_max);
	}

	// Finds the closest intersection
	template <typename ISECT = SimpleIsect>
	bool intersect(const Ray& ray, ISECT& out_result) {
		return geometry_->intersect(ray.o, ray.d, out_result);
	}

	AbstractGeometry* getGeometry() const { return geometry_;}
	std::shared_ptr<Camera> getCamera() const { return camera_; }
	const Medium& getMedium() const { return medium_; }
};

class SceneBuilder{
	typedef HostMedium Medium;

	public:
		virtual Camera* getCamera() = 0;
        virtual AbstractGeometry* getGeometry() = 0;
		virtual Medium getMedium() = 0;
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
				builder->getCamera(),
				builder->getGeometry(),
				builder->getMedium()
				);
		return scene;
	}
};



#endif /* SCENE_H_ */
