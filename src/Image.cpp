#include <iostream>
#include <string>
#include <stb_image_write.h>

#include "Image.h"


Image::Image(int x, int y) :
        xSize(x),
        ySize(y)
        {
	pixels = (float4 *)malloc(x * y * sizeof(float4));
}

Image::~Image() {
    free(pixels);
}

void Image::setPixel(int x, int y, const float4 &pixel) {
    assert(x >= 0 && y >= 0 && x < xSize && y < ySize);
    pixels[(y * xSize) + x] = pixel;
}

void Image::savePNG(const std::string &baseFilename) const{
    unsigned char *bytes = new unsigned char[4 * xSize * ySize];
    for (int y = 0; y < ySize; y++) {
        for (int x = 0; x < xSize; x++) {
            int i = y * xSize + x;
            float4 pix = clamp(pixels[i], 0, 1) * 255.f;
            //float3 pix = make_float3(clamp(pixels[i].x)*255.f, clamp(pixels[i].x)*255.f, clamp(pixels[i].x)*255.f);
            bytes[3 * i + 0] = (unsigned char) pix.x;
            bytes[3 * i + 1] = (unsigned char) pix.y;
            bytes[3 * i + 2] = (unsigned char) pix.z;
        }
    }

    std::string filename = baseFilename + ".png";
    stbi_write_png(filename.c_str(), xSize, ySize, 3, bytes, xSize * 3);
    std::cout << "Saved " << filename << "." << std::endl;

    delete[] bytes;
}

void Image::saveHDR(const std::string &baseFilename) const{
    std::string filename = baseFilename + ".hdr";
    stbi_write_hdr(filename.c_str(), xSize, ySize, 4, (const float *) pixels);
    std::cout << "Saved " + filename + "." << std::endl;
}
