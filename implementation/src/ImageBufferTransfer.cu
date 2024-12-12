#include "ImageBufferTransfer.h"
#include "Utilities.h"

template <typename T_IN, typename T_OUT, typename FUNC>
__global__  void d_transform(T_IN* buffer_in, T_OUT* buffer_out, unsigned int n, FUNC transform) {

	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (x > n) return;

	//apply transformation
	buffer_out[x] = transform(buffer_in[x]);
}


template <typename T_IN, typename T_OUT, typename FUNC>
__global__  void d_accumulateAndTransform2D(T_IN* buffer_in, size_t pitch_in, T_IN* buffer_transfer, T_OUT* buffer_out, size_t pitch_out, unsigned int width, unsigned int height, FUNC transform)
{
	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x > width || y > height) return;

	int index_out = (y * pitch_out) + x;
	int index_in = (y * pitch_in) + x;

	buffer_transfer[index_out] += buffer_in[index_in];

	//apply transformation
	buffer_out[index_out] = transform(buffer_transfer[index_out]);
}

void HostImageBufferTansferDelegate::transfer(void* buffer_in, size_t pitch_in, void* _buffer_out, size_t _offset_out, size_t pitch_out, size_t width, size_t height, UtilityFunctors::Scale transform) {
	
	int offset_out =  _offset_out / sizeof(float);
	float* buffer_out = (float*)_buffer_out + offset_out;

	dim3 block_size(32);
	uint n = width*height / sizeof(float);
	dim3 grid_size = dim3(divUp(n, block_size.x));

	d_transform << <grid_size, block_size >> >((float*)buffer_in, (float*)buffer_in, n, transform);
	checkCudaErrors(cudaMemcpy2DAsync(buffer_out, pitch_out, buffer_in, pitch_in, width, height, cudaMemcpyDeviceToHost));
};

template <typename FUNC>
struct ColorPixelTransform
{
	FUNC transform;
	__host__ __device__ ColorPixelTransform(FUNC _transform):transform(_transform){}

	__host__ __device__
		uchar1 operator()(float& pixel)
	{
		uchar1 c;
		c.x = clamp(powf(transform(pixel), 1.f / 2.2f) * 255.f, 0.f, 255.f);
		return c;
	}
};

void DeviceImageBufferTansferDelegate::transfer(void* buffer_in, size_t pitch_in, void* buffer_out, size_t _offset_out, size_t pitch_out, size_t width, size_t height, UtilityFunctors::Scale transform) {

	dim3 block_size(32);
	uint n = width*height / sizeof(float);
	dim3 grid_size = dim3(divUp(n, block_size.x));

	d_transform << <grid_size, block_size >> >((float*)buffer_in, (uchar1*)buffer_out, n, ColorPixelTransform<UtilityFunctors::Scale>(transform));
}


void DeviceTiledImageBufferTansferDelegate::transfer(void* buffer_in, size_t pitch_in, void* _buffer_out, size_t _offset_out, size_t pitch_out, size_t width, size_t height, UtilityFunctors::Scale transform) {


	int offset_out = _offset_out / sizeof(float);
	float* d_buffer_transfer_ptr = d_buffer_transfer_ + offset_out;

	uchar1* buffer_out = (uchar1*)_buffer_out + offset_out;

	dim3 block_size(32,32);
	int n_floats_in_a_row = width / sizeof(float);
	dim3 grid_size = dim3(divUp(n_floats_in_a_row, block_size.x), divUp(height, block_size.y));

	// facility trick which permits to reset the buffer when the camera is reset (should change in the future)
	if (transform.scale == 1) {
		cudaMemset2D(d_buffer_transfer_ptr, pitch_out, 0, width, height);
	}

	d_accumulateAndTransform2D <<< grid_size, block_size >>>
		(
			(float*)buffer_in,
			pitch_in/sizeof(float),
			d_buffer_transfer_ptr,
			(uchar1*)buffer_out,
			pitch_out/sizeof(float),
			n_floats_in_a_row,
			height,
			ColorPixelTransform<UtilityFunctors::Scale>(transform)
		);
};