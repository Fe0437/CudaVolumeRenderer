#ifndef INTEGRATOR_H_
#define INTEGRATOR_H_

#include <cuda_runtime.h>

#include "Buffer.h"

class AbstractRenderer {
 public:
  virtual ~AbstractRenderer() = default;
  virtual void render(Buffer2D buffer_out) = 0;
};

class AbstractProgressiveRenderer : public AbstractRenderer {
 public:
  ~AbstractProgressiveRenderer() override = default;
  using uint = unsigned int;

  virtual void initRendering() = 0;
  virtual void runIterations() = 0;
  virtual void setNIterations(uint n_iterations) = 0;
  virtual bool imageComplete() = 0;
  virtual void getImage(Buffer2D buffer_out) = 0;
};

template <typename FUNC>
class Buffer2DTransferDelegate {
 public:
  virtual void transfer(Buffer2D buffer_in, Buffer2D buffer_out,
                        uint2 offset_out, FUNC transform) = 0;
};

#endif /* INTEGRATOR_H_ */
