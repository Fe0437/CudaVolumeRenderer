#ifndef CUDAINTERACTIVERENDERER_H_
#define CUDAINTERACTIVERENDERER_H_
#pragma once

// GLEW must be included before any OpenGL headers
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <fstream>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "AbstractRenderer.h"
#include "Buffer.h"
#include "Config.h"
#include "helper_cuda.h"
#include "helper_math.h"

class InputController {
 protected:
  std::function<void(void)> input_callback_;

 public:
  virtual void setInputCallback(std::function<void(void)> fp) {
    input_callback_ = fp;
  }

  virtual void init(GLFWwindow* gl_view_);
  virtual void mouseButtonCallback(GLFWwindow* window, int button, int action,
                                   int mods) {};
  virtual void mousePositionCallback(GLFWwindow* window, double xpos,
                                     double ypos) {};
  virtual void keyCallback(GLFWwindow* window, int key, int scancode,
                           int action, int mods) {};

  virtual void update() = 0;
};

//-------------------- OOP modification for glfw--------------------------
void _mouseButtonCallback(GLFWwindow* window, int button, int action,
                          int mods) {
  auto* input = (InputController*)glfwGetWindowUserPointer(window);
  input->mouseButtonCallback(window, button, action, mods);
}

void _mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
  auto* input = (InputController*)glfwGetWindowUserPointer(window);
  input->mousePositionCallback(window, xpos, ypos);
}

void _keyCallback(GLFWwindow* window, int key, int scancode, int action,
                  int mods) {
  auto* input = (InputController*)glfwGetWindowUserPointer(window);
  input->keyCallback(window, key, scancode, action, mods);
}

void InputController::init(GLFWwindow* gl_view_) {
  glfwSetWindowUserPointer(gl_view_, this);
  glfwSetCursorPosCallback(gl_view_, _mousePositionCallback);
  glfwSetMouseButtonCallback(gl_view_, _mouseButtonCallback);
}
//--------------------------------------------------------------------------------

class BufferProcessorDelegate {
 public:
  virtual void init(uint2 resolution, GLuint pbo_Id_) = 0;
  virtual void reset() = 0;
  virtual void process() = 0;
};

class GLViewController {
  GLFWwindow* gl_view_{};
  std::string view_name_{"Volumetric Path Tracing"};
  int width_{1};
  int height_{1};
  std::unique_ptr<InputController> input_controller_{};

  GLuint displayed_image_id_{};
  GLuint pbo_Id_{};

  std::unique_ptr<BufferProcessorDelegate> pbo_delegate_{};

 public:
  GLViewController(int width, int height)
      : width_(width),
        height_(height),
        view_name_("GL_VIEW"),
        pbo_delegate_{} {}

  virtual ~GLViewController() = default;

  void setBufferProcessorDelegate(
      std::unique_ptr<BufferProcessorDelegate>&& pbo_delegate) {
    pbo_delegate_ = std::move(pbo_delegate);
  }

  void setInputController(std::unique_ptr<InputController>&& input_controller) {
    input_controller_ = std::move(input_controller);
    input_controller_->setInputCallback([this]() { reset(); });
  }

  void init() {
    glDisable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT);

    initGLView();
    initPixelBufferObject();
    initTexture();
    if (input_controller_) input_controller_->init(gl_view_);
    if (pbo_delegate_)
      pbo_delegate_->init(make_uint2(width_, height_), pbo_Id_);
  }

  void reset() {
    if (pbo_delegate_) pbo_delegate_->reset();
  }

  /*create a pixel buffer object*/
  void initPixelBufferObject() {
    // set up vertex data parameter
    int num_texels = width_ * height_;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo_Id_);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_Id_);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
  }

  void initTexture() {
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &displayed_image_id_);
    glBindTexture(GL_TEXTURE_2D, displayed_image_id_);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_BGRA,
                 GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  }

  void initGLView() {
    glfwSetErrorCallback([](int error, const char* description) {
      fprintf(stderr, "%s\n", description);
    });

    if (!glfwInit()) {
      exit(EXIT_FAILURE);
    }

    gl_view_ =
        glfwCreateWindow(width_, height_, view_name_.c_str(), NULL, NULL);
    if (!gl_view_) {
      glfwTerminate();
      return;
    }

    glfwMakeContextCurrent(gl_view_);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
      return;
    }
  }

  void drawTexture() {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_Id_);
    glBindTexture(GL_TEXTURE_2D, displayed_image_id_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA,
                    GL_UNSIGNED_BYTE, NULL);

    glBegin(GL_QUADS);
    glTexCoord2f(0, 1.0F);
    glVertex3f(-1.F, -1.F, 0);

    glTexCoord2f(0, 0);
    glVertex3f(0 - 1.F, 1.0F, 0);

    glTexCoord2f(1.0F, 0);
    glVertex3f(1.0F, 1.0F, 0);

    glTexCoord2f(1.0F, 1.0F);
    glVertex3f(1.0F, -1.F, 0);
    glEnd();

    glfwSwapBuffers(gl_view_);
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(gl_view_)) {
      glfwPollEvents();
      input_controller_->update();
      pbo_delegate_->process();
      glfwSetWindowTitle(gl_view_, view_name_.c_str());
      drawTexture();
    }
    glfwDestroyWindow(gl_view_);
    glfwTerminate();
  }
};

class CameraController : public InputController {
  std::shared_ptr<Camera> camera_;
  bool left_mouse_pressed_;
  bool right_mouse_pressed_;
  bool middle_mouse_pressed_;
  double last_x_;
  double last_y_;
  bool dirty_flag_ = false;
  static constexpr float rotation_velocity{1.F};

 public:
  explicit CameraController(const std::shared_ptr<Camera>& camera)
      : last_x_(0),
        last_y_(0),
        left_mouse_pressed_(false),
        right_mouse_pressed_(false),
        middle_mouse_pressed_(false) {
    camera_ = camera;
  }

  void mouseButtonCallback(GLFWwindow* window, int button, int action,
                           int mods) {
    left_mouse_pressed_ =
        (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    right_mouse_pressed_ =
        (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    middle_mouse_pressed_ =
        (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
  }

  void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (xpos == last_x_ && ypos == last_y_) {
      return;
    }

    double dx;
    double dy;
    dx = (float)(xpos - last_x_);
    dy = (float)(ypos - last_y_);

    auto resolution = camera_->getResolution();

    if (left_mouse_pressed_) {
      // left == rotate
      float dtheta = dy / resolution.y;
      float dphi = dx / resolution.x;
      camera_->lazyRotateAroundTheCenterBy(rotation_velocity * dtheta,
                                           rotation_velocity * dphi);
      dirty_flag_ = true;
    } else if (right_mouse_pressed_) {
      // right = zoom
      float dzoom = fmaxf(dx / resolution.x, dy / resolution.y);
      camera_->lazyMoveBy(0, 0, dzoom);
      dirty_flag_ = true;
    } else if (middle_mouse_pressed_) {
      // middle = translate
      float _y = dy / resolution.y;
      float _x = dx / resolution.x;
      camera_->lazyMoveBy(_x, _y, 0);
      dirty_flag_ = true;
    }
    last_x_ = xpos;
    last_y_ = ypos;
  }

  void update() override {
    if (dirty_flag_) {
      camera_->lazyUpdate();
      input_callback_();
      dirty_flag_ = false;
    }
  }
};

#include <ctime>

class CudaInteractiveRenderer : public BufferProcessorDelegate {
  std::unique_ptr<AbstractProgressiveRenderer> renderer_{};
  GLuint pbo_id_{};
  cudaGraphicsResource* cuda_pbo_resource_{};
  uint n_iter_per_frame_{1};
  uint current_iter_{};
  const float max_elapsed_time{1.F / 10.F};
  uint max_iter_{1000000000};
  uint2 resolution_{};

 public:
  explicit CudaInteractiveRenderer(
      std::unique_ptr<AbstractProgressiveRenderer> renderer)
      : renderer_(std::move(renderer)) {
    renderer_->setNIterations(n_iter_per_frame_);
  }

  ~CudaInteractiveRenderer() = default;

  void init(uint2 resolution, GLuint pbo_Id) {
    pbo_id_ = pbo_Id;
    resolution_ = resolution;
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(
        &cuda_pbo_resource_, pbo_id_, cudaGraphicsMapFlagsWriteDiscard));
    reset();
  };

  void reset() override {
    renderer_->initRendering();
    current_iter_ = 0;
  }

  void process() override {
    if (current_iter_ < max_iter_) {
      void* d_ptr = nullptr;
      size_t num_bytes = 0;

      // Map the PBO for CUDA
      checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource_, 0));
      checkCudaErrors(cudaGraphicsResourceGetMappedPointer(&d_ptr, &num_bytes,
                                                           cuda_pbo_resource_));
      assert(num_bytes == resolution_.x * resolution_.y * sizeof(uchar4));

      clock_t t_start = clock();
      float elapsed{};

      Buffer2D image_buffer = make_buffer2D<uchar4>(
          reinterpret_cast<uchar4*>(d_ptr), resolution_.x, resolution_.y);
      do {
        renderer_->runIterations();
        renderer_->getImage(image_buffer);

        if (renderer_->imageComplete()) current_iter_ += n_iter_per_frame_;

        elapsed = static_cast<float>(clock() - t_start) / CLOCKS_PER_SEC;

      } while (current_iter_ == 0 || elapsed <= max_elapsed_time);

      cudaDeviceSynchronize();
      // Unmap the resource
      checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource_, 0));
    }
  }
};

#endif  // !CUDAINTERACTIVERENDERER_H_
