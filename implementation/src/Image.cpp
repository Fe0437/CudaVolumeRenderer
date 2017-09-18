#include "Image.h"

#include <stb_image_write.h>

#include <vector>

Image::Image(int x, int y) : width_(x), height_(y) {
#ifdef __CUDA_RUNTIME_H__
  auto ret = cudaHostAlloc((void **)&pixels, x * y * sizeof(float4),
                           cudaHostAllocWriteCombined);
  if (ret != cudaSuccess) throw(" Error : impossible to allocate image memory");
#else
  pixels = (float4 *)malloc(x * y * sizeof(float4));
#endif
}

Image::~Image() noexcept {
#ifdef __CUDA_RUNTIME_H__
  cudaDeviceSynchronize();
  try {
    auto ret = cudaFreeHost((void *)pixels);
    if (ret != cudaSuccess) {
      // Log error instead of throwing in destructor
      std::cerr << "Error: impossible to release image memory" << std::endl;
    }
  } catch (...) {
    // Log any unexpected errors
    std::cerr << "Unexpected error in Image destructor" << std::endl;
  }
#else
  free(pixels);
#endif
}

void Image::savePNG(const std::string &baseFilename) const {
  // Use vector for automatic memory management
  std::vector<unsigned char> bytes(3 * width_ * height_);

  // Use structured bindings and range-based for loop
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      const auto i = (y * width_) + x;
      const auto [r, g, b, a] =
          static_cast<float4>(clamp(pixels[i], 0, 1) * 255.f);

      bytes[(3 * i) + 0] = static_cast<unsigned char>(r);
      bytes[(3 * i) + 1] = static_cast<unsigned char>(g);
      bytes[(3 * i) + 2] = static_cast<unsigned char>(b);
    }
  }

  const auto filename = baseFilename + ".png";
  stbi_write_png(filename.c_str(), width_, height_, 3, bytes.data(),
                 width_ * 3);
  COUT_DEBUG("Saved " << filename << ".")
}

void Image::saveHDR(const std::string &baseFilename) const {
  std::string filename = baseFilename + ".hdr";
  stbi_write_hdr(filename.c_str(), width_, height_, 4, (const float *)pixels);
  COUT_DEBUG("Saved " + filename + ".")
}
