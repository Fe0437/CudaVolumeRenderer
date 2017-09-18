/*
 * Camera.h
 *
 *  Created on: 30/ago/2017
 *      Author: macbook
 */

#ifndef CAMERA_H_
#define CAMERA_H_

#ifdef WIN32
#include <windows.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include "Utilities.h"
#include "helper_cuda.h"

//cuda
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace std;

class CudaSensor{

	glm::ivec2 resolution_;
	float4* d_pixel_buffer_;

public:
	CudaSensor(glm::ivec2 resolution):
	resolution_(resolution),
	d_pixel_buffer_(0)
	{}

	bool initBuffer(){
		cudaMalloc((void **)&d_pixel_buffer_, resolution_.x*resolution_.y*sizeof(float4));
		cudaMemset(d_pixel_buffer_, 0, resolution_.x*resolution_.y*sizeof(float4));
		//CHECK_CUDA_ERROR("initialized pixel buffer used by CUDA for rendering");
		return true;
	}

	void* getBuffer(){ return d_pixel_buffer_;}
	glm::ivec2 getResolution(){ return resolution_;}

};


class RenderCamera {

	CudaSensor sensor_;
	glm::vec3 position_;
	glm::vec3 view_;
	glm::vec3 up_;
	glm::vec2 fov_;
	glm::mat4 inv_view_matrix_;

public:

	RenderCamera():
	up_(glm::vec3(0,1,0)),
	view_(glm::vec3(0,0,1)),
	sensor_(glm::vec2(512,512))
	{
		sensor_.initBuffer();
	}

	RenderCamera(const glm::vec2& resolution,const  glm::vec3& view, const glm::vec3& up,const  glm::vec3& position, float fovy):
	up_(up),
	view_(view),
	position_(position),
	sensor_(glm::vec2(512,512))
	{
		setFovFromY(fovy);
	}

	void calculateInvViewMatrixFromBasis(){
		glm::vec3 r = glm::cross(view_, up_);
		inv_view_matrix_[0] = glm::vec4(r,0);
		inv_view_matrix_[1] = glm::vec4(up_, 0);
		inv_view_matrix_[2] = glm::vec4(view_, 0);
		inv_view_matrix_[3] = glm::vec4(position_, 1.0);
	}

	glm::mat4 getInvViewMatrix(){
		assert(inv_view_matrix_!= glm::mat4());
		return inv_view_matrix_;
	}

	//calculate fov based on resolution
	void setFovFromY(float fovy){
		glm::ivec2 resolution = sensor_.getResolution();
		float yscaled = tan(fovy * (PI / 180));
		float xscaled = (yscaled * resolution.x) / resolution.y;
		float fovx = (atan(xscaled) * 180) / PI;
		fov_ = glm::vec2(fovx, fovy);
	}

	//rotate the camera
	void rotateTo(float theta, float phi)
	{
		glm::vec3 v = view_;
		glm::vec3 u = up_;
		glm::vec3 r = glm::cross(v, u);
		glm::mat4 rotmat = glm::rotate(theta, r) * glm::rotate(phi, u);
		view_ = glm::vec3(rotmat * glm::vec4(v, 0.f));
		up_ = glm::vec3(rotmat * glm::vec4(u, 0.f));
	}

	//move the camera
	void moveBy(glm::vec3 dist)
		{
			glm::vec3 r = glm::cross(view_, up_);
			position_ += dist.x * glm::cross(view_, up_) + dist.y * up_ + dist.z * view_;
		}

	virtual ~RenderCamera() = default;
};

#endif /* CAMERA_H_ */
