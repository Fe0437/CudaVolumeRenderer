/*
 * Camera.h
 *
 *  Created on: 30/ago/2017
 *      Author: Federico Forti
 */

#ifndef CAMERA_H_
#define CAMERA_H_
#pragma once

#include "Math.h"
//#include <GLFW/glfw3.h>

class Camera {

	glm::ivec2 resolution_;
	glm::vec2 fov_;
	glm::mat4 model_view_matrix_;
	glm::vec2 lazy_view_rotation_;
	glm::vec3 lazy_view_translation_;

public:

	Camera(int res_x=400, int res_y=400, float fov_x=0.7):
	resolution_(res_x, res_y)
	{
		setFovFromX(fov_x);

#ifdef MITSUBA_COMPARABLE
		model_view_matrix_[0] = glm::vec4(1, 0, 0, 0); //right
#else
		model_view_matrix_[0] = glm::vec4(-1, 0, 0, 0); //right
#endif
		model_view_matrix_[1] = glm::vec4(0, -1, 0, 0); //up
		model_view_matrix_[2] = glm::vec4(0, 0, -1, 0); //view
		model_view_matrix_[3] = glm::vec4(0, 0, 100.0f, 1); //position

		lazy_view_translation_ = getPosition();
	}

	glm::vec3 getPosition() { return glm::vec3(model_view_matrix_[3]); }
	glm::vec3 getUp() { return glm::vec3(model_view_matrix_[1]); }
	glm::vec3 getRight() { return glm::vec3(model_view_matrix_[0]); }
	glm::vec3 getView() { return glm::vec3(model_view_matrix_[2]); }

	glm::ivec2 getResolution() { return resolution_; }
	void setResolution(int x, int y) { 
		resolution_ = { x,y };
		setFovFromX(fov_.x);
	}

	/*Camera(const uint2& resolution,const  float3& view, const float3& up,const  float3& position, float fovx):
	resolution_(resolution_.x, resolution_.y)
	{
		glm::vec3 _up(up.x, up.y, up.z);
		glm::vec3 _view(view.x, view.y, view.z);
		glm::vec3 _right = -glm::cross(_up, _view);
		glm::vec3 _position(position.x, position.y, position.z);

		model_view_matrix_[0] = glm::vec4(_right, 0);
		model_view_matrix_[1] = glm::vec4(_up, 0);
		model_view_matrix_[2] = glm::vec4(_view, 0);
		model_view_matrix_[3] = glm::vec4(_position, 1.0);
		lazy_view_translation_ = _position;
		setFovFromX(fovx);
	}*/

	virtual ~Camera() = default;

	const float* getModelViewMatrix() const {
		assert(model_view_matrix_!= glm::mat4());
		return  glm::value_ptr(model_view_matrix_);
	}

	//calculate fov based on resolution
	void setFovFromX(float fovx) {
		fov_.x = fovx;
		fov_.y = ((float)resolution_.y / (float)resolution_.x) *fov_.x;
		//LOG_DEBUG("fov_x %f fov_y %f \n", fov_.x, fov_.y);
	}

	float2 getRasterToView() const{
		return  make_float2(tan(fov_.x * PI / 360.f), tan(fov_.y * PI / 360.f));
	}

	//rotate the camera
	void lazyRotateAroundTheCenterBy(float _dtheta, float _dphi)
	{
		lazy_view_rotation_.x += _dtheta;
		lazy_view_rotation_.y += _dphi;
	}

	void lazyMoveBy(float x,float y,float z)
	{
		lazy_view_translation_ += glm::vec3(x, y, z);
	}

	void lazyUpdate()
	{
		model_view_matrix_ = -glm::rotate(-lazy_view_rotation_.x, glm::vec3(1, 0, 0));
		model_view_matrix_ *= glm::rotate(-lazy_view_rotation_.y, glm::vec3(0, 1, 0));
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
		glTranslatef(lazy_view_translation_.x, lazy_view_translation_.y, lazy_view_translation_.z);
		glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(model_view_matrix_));
		glPopMatrix();
		*/
		
	}

private:

};

#endif /* CAMERA_H_ */
