/*
 * Defines.h
 *
 *  Created on: 28/ago/2017
 *      Author: macbook
 */

#ifndef DEFINES_H_
#define DEFINES_H_

#ifdef WIN32
#include <windows.h>
#endif


#define INV_FOURPI 0.0796
#define INV_PI 0.3184

#define PI                3.1415926535897932384626422832795028841971f
#define TWOPI            6.2831853071795864769252867665590057683943f
#define EPSILON           0.00001f
#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

#define COLOR3 float3
#define COLOR4 float4
#define FLOAT float

#define FLOAT3 float3

#define ALBEDO_T COLOR4


#endif /* DEFINES_H_ */
