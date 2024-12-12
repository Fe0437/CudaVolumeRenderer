
#include <stb_image_write.h>
#include "Image.h"


Image::Image(int x, int y) :
        width_(x),
        height_(y)
        {

#ifdef __CUDA_RUNTIME_H__
			auto ret = cudaHostAlloc((void**)&pixels, x * y * sizeof(float4), cudaHostAllocWriteCombined);
			if( ret != cudaSuccess)  throw(" Error : impossible to allocate image memory");
#else
			pixels = (float4 *)malloc(x * y * sizeof(float4));
#endif
		}

Image::~Image() {
#ifdef __CUDA_RUNTIME_H__
	cudaDeviceSynchronize();
	auto ret = cudaFreeHost((void*)pixels);
	if (ret != cudaSuccess)  throw(" Error : impossible to release image memory");

#else
    free(pixels);
#endif
}


void Image::savePNG(const std::string &baseFilename) const{
    unsigned char *bytes = new unsigned char[4 * width_ * height_];
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            int i = y * width_ + x;
            float4 pix = clamp(pixels[i], 0, 1) * 255.f;
            bytes[3 * i + 0] = (unsigned char) pix.x;
            bytes[3 * i + 1] = (unsigned char) pix.y;
            bytes[3 * i + 2] = (unsigned char) pix.z;
        }
    }

    std::string filename = baseFilename + ".png";
    stbi_write_png(filename.c_str(), width_, height_, 3, bytes, width_ * 3);
    COUT_DEBUG ("Saved " << filename << ".")
    delete[] bytes;
}

void Image::saveHDR(const std::string &baseFilename) const{
    std::string filename = baseFilename + ".hdr";
    stbi_write_hdr(filename.c_str(), width_, height_, 4, (const float *) pixels);
	COUT_DEBUG("Saved " + filename + ".")
}
