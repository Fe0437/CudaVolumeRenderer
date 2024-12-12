/*
 * Integrator.h
 *
 *  Created on: 30/ago/2017
 *      Author: macbook
 */

#ifndef INTEGRATOR_H_
#define INTEGRATOR_H_

#include <cuda_runtime.h>

class AbstractRenderer
{
public:
	virtual ~AbstractRenderer() {}
	virtual void render(void* buffer_out) = 0;
};

class AbstractProgressiveRenderer : public AbstractRenderer {

public:
	virtual ~AbstractProgressiveRenderer() {}
	typedef unsigned int uint;

	virtual void initRendering() = 0;
	virtual void runIterations() = 0;
	virtual void setNIterations(uint n_iterations) = 0;
	virtual bool imageComplete() = 0;
	virtual void getImage(void* buffer_out) = 0;
};

template <typename FUNC>
class Buffer2DTransferDelegate {
public:
	virtual void transfer(void* buffer_in, size_t pitch_in, void* buffer_out, size_t offset_out, size_t pitch_out, size_t width, size_t height, FUNC transform)=0;
};



#endif /* INTEGRATOR_H_ */
