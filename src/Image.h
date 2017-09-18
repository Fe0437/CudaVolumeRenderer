#ifndef IMAGE_H_
#define IMAGE_H_

#ifdef WIN32
#include <windows.h>
#endif

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "helper_math.h"


using namespace std;

class Image {
private:
    int xSize;
    int ySize;

public:
    float4 *pixels;

    Image(int x, int y);
    ~Image();
    glm::ivec2 getResolution() const { return glm::ivec2(xSize, ySize); }
    void setPixel(int x, int y, const float4 &pixel);
    void savePNG (const std::string &baseFilename) const;
    void saveHDR (const std::string &baseFilename) const;
};


#endif /* IMAGE_H_ */
