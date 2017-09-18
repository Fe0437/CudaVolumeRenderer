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

#include "Camera.h"
#include "Defines.h"
#include "Geometry.h"
#include "Medium.h"

class Scene {
  typedef HostMedium Medium;

 private:
  std::shared_ptr<AbstractGeometry> geometry_;
  std::shared_ptr<Camera> camera_;
  Medium medium_{};

 public:
  Scene(std::shared_ptr<Camera> camera,
        std::shared_ptr<AbstractGeometry> geometry, Medium medium)
      : camera_(std::move(camera)),
        geometry_(std::move(geometry)),
        medium_(std::move(medium)) {}

  Scene() = default;
  ~Scene() = default;

  void growBBox(float3& box_min, float3& box_max) {
    geometry_->growBBox(box_min, box_max);
  }

  // Finds the closest intersection
  template <typename ISECT = SimpleIsect>
  auto intersect(const Ray& ray, ISECT& out_result) -> bool {
    return geometry_->intersect(ray.o, ray.d, out_result);
  }

  [[nodiscard]] auto getGeometry() const -> std::shared_ptr<AbstractGeometry> {
    return geometry_;
  }
  [[nodiscard]] auto getCamera() const -> std::shared_ptr<Camera> {
    return camera_;
  }
  [[nodiscard]] auto getMedium() const -> const Medium& { return medium_; }
};

class SceneBuilder {
  typedef HostMedium Medium;

 public:
  virtual auto getCamera() -> std::shared_ptr<Camera> = 0;
  virtual auto getGeometry() -> std::shared_ptr<AbstractGeometry> = 0;
  virtual auto getMedium() -> Medium = 0;
};

class SceneAssembler {
  std::unique_ptr<SceneBuilder> builder{};

 public:
  SceneAssembler() = default;
  virtual ~SceneAssembler() = default;

  void setBuilder(std::unique_ptr<SceneBuilder> newBuilder) {
    builder = std::move(newBuilder);
  }

  auto getScene() -> Scene {
    Scene scene(builder->getCamera(), builder->getGeometry(),
                builder->getMedium());
    return scene;
  }
};

#endif /* SCENE_H_ */
