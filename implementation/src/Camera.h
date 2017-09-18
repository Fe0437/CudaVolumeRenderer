/*
 * Camera.h
 *
 *  Created on: 30/ago/2017
 *      Author: Federico Forti
 */

#ifndef CAMERA_H_
#define CAMERA_H_
#pragma once

#include <glm/gtc/quaternion.hpp>

#include "CVRMath.h"

class Camera {
  glm::ivec2 resolution_{};
  glm::vec2 fov_{};
  glm::mat4 model_view_matrix_{};
  glm::vec2 lazy_view_rotation_{};
  glm::vec3 lazy_view_translation_{};
  glm::quat orientation_{};

 public:
  explicit Camera(int res_x = 400, int res_y = 400, float fov_x = 0.7)
      : resolution_(res_x, res_y) {
    setFovFromX(fov_x);

    // First set up the model view matrix
#ifdef MITSUBA_COMPARABLE
    model_view_matrix_[0] = glm::vec4(1, 0, 0, 0);  // right
#else
    model_view_matrix_[0] = glm::vec4(-1, 0, 0, 0);  // right
#endif
    model_view_matrix_[1] = glm::vec4(0, -1, 0, 0);      // up
    model_view_matrix_[2] = glm::vec4(0, 0, -1, 0);      // view
    model_view_matrix_[3] = glm::vec4(0, 0, 100.0f, 1);  // position

    // Initialize orientation from model view matrix
    orientation_ = glm::quat_cast(glm::mat3(model_view_matrix_));
    lazy_view_translation_ = getPosition();
  }

  glm::vec3 getPosition() { return glm::vec3(model_view_matrix_[3]); }
  glm::vec3 getUp() { return glm::vec3(model_view_matrix_[1]); }
  glm::vec3 getRight() { return glm::vec3(model_view_matrix_[0]); }
  glm::vec3 getView() { return glm::vec3(model_view_matrix_[2]); }

  glm::ivec2 getResolution() { return resolution_; }
  void setResolution(int x, int y) {
    resolution_ = {x, y};
    setFovFromX(fov_.x);
  }

  virtual ~Camera() = default;

  [[nodiscard]] auto getModelViewMatrix() const -> const float* {
    assert(model_view_matrix_ != glm::mat4());
    return glm::value_ptr(model_view_matrix_);
  }

  // calculate fov based on resolution
  void setFovFromX(float fovx) {
    fov_.x = fovx;
    fov_.y = ((float)resolution_.y / (float)resolution_.x) * fov_.x;
    // LOG_DEBUG("fov_x %f fov_y %f \n", fov_.x, fov_.y);
  }

  [[nodiscard]] float2 getRasterToView() const {
    return make_float2(tan(fov_.x * PI / 360.f), tan(fov_.y * PI / 360.f));
  }

  // rotate the camera
  void lazyRotateAroundTheCenterBy(float _dtheta, float _dphi) {
    glm::quat q_pitch = glm::angleAxis(_dtheta, glm::vec3(1, 0, 0));
    glm::quat q_yaw = glm::angleAxis(_dphi, glm::vec3(0, 1, 0));
    orientation_ = glm::normalize(q_pitch * orientation_ * q_yaw);
  }

  void lazyMoveBy(float x, float y, float z) {
    // Make z movement (zoom) more responsive
    const float ZOOM_SPEED_MULTIPLIER = 5.0F;
    lazy_view_translation_ += glm::vec3(x, y, z * ZOOM_SPEED_MULTIPLIER);
  }

  void lazyUpdate() {
    model_view_matrix_ = glm::mat4_cast(orientation_);
    model_view_matrix_ *= glm::translate(-lazy_view_translation_);

#ifdef MITSUBA_COMPARABLE
    model_view_matrix_[0] *= -1;
#endif
    /*
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glRotatef(-lazy_view_rotation_.x, 1.0, 0.0, 0.0);
    glRotatef(-lazy_view_rotation_.y, 0.0, 1.0, 0.0);
    glTranslatef(lazy_view_translation_.x, lazy_view_translation_.y,
    lazy_view_translation_.z); glGetFloatv(GL_MODELVIEW_MATRIX,
    glm::value_ptr(model_view_matrix_)); glPopMatrix();
    */
  }

  // Add lookAt function
  void lookAt(const glm::vec3& eye, const glm::vec3& center,
              const glm::vec3& up) {
    glm::vec3 forward = glm::normalize(center - eye);
    glm::vec3 right = glm::normalize(glm::cross(forward, up));
    glm::vec3 new_up = glm::normalize(glm::cross(right, forward));

    model_view_matrix_[0] = glm::vec4(right, 0.0f);
    model_view_matrix_[1] = glm::vec4(new_up, 0.0f);
    model_view_matrix_[2] = glm::vec4(-forward, 0.0f);
    model_view_matrix_[3] = glm::vec4(eye, 1.0f);

    // Update lazy translation to match new position
    lazy_view_translation_ = eye;

    // Reset orientation
    orientation_ = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  }
};

#endif /* CAMERA_H_ */
