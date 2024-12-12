#ifndef IMAGE_H_
#define IMAGE_H_

#include "Defines.h"
#include "Math.h"
#include <iostream>
#include <string>


class Image {
private:

    int width_;
    int height_;
public:

	float4 *pixels;
	Image():width_(0),height_(0){};
    Image(int x, int y);
    ~Image();
    uint2 getResolution() const { return make_uint2(width_, height_); }
    void savePNG (const std::string &baseFilename) const;
    void saveHDR (const std::string &baseFilename) const;
};


#endif /* IMAGE_H_ */
