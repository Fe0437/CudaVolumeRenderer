/*
 * Defines.h
 *
 *  Created on: 28/ago/2017
 *      Author: macbook
 */

#ifndef DEFINES_H_
#define DEFINES_H_

#include <cuda_runtime.h>


//----------------------FORCING MAXIMUM OCCUPANCY-------------------------
//#define MAXIMAZE_OCCUPANCY  //uncomment this to maximaze occupancy

#define MAX_THREADS_PER_BLOCK 1024
#define STREAMING_THREADS_BLOCK MAX_THREADS_PER_BLOCK
#define MIN_BLOCKS_PER_MULTIPROCESSOR 2

#ifdef MAXIMAZE_OCCUPANCY
#define KERNEL_LAUNCH __global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR)
#else
#define KERNEL_LAUNCH __global__ void 
#endif
//------------------------------------------------------------------------

//----------------------------PROGRAM SETTINGS------------------------------

#define STREAMING_ITEMS_PER_THREAD 1 //<number of items for thread for the streaming algorithm

#define NAIVE_MK_COMPACTION //< if it is enable the compaction is enabled in the naiveMK method
#define REGENERATION_SYNCHRONIZATION_LEVEL 0 //< 0 : single thread regeneration, 1: warp regeneration, 2 : block regeneration

#define RUSSIAN_ROULETTE //<if it is enabled the Russian Roulette is enabled in all the methods

#define MHA_SUPPORT // enable MHA and MHD file support
#define MITSUBA_COMPARABLE    //<if it is enabled the result of the rendering can be compared to the mitsuba renderer
#define RAYS_STATISTICS		//<if it is enabled the number of rays traced is counted and more statistics are provided
#define DOUBLE_BUFFERING		//<if it is enabled a double buffer is used for transfering image data
//------------------------------------------------------------------------

//--------------------------------CONSTANT----------------------------

//kernel constant memory used
#define STREAMING_SHARED_MEMORY 37072

#define INV_FOURPI 0.0796
#define INV_PI 0.3184

#define PI                3.1415926535897932384626422832795028841971f
#define TWOPI            6.2831853071795864769252867665590057683943f
#define EPSILON           0.00001f
#define DENOM_EPS		  EPSILON
//------------------------------------------------------------------------


//--------------------------------DEBUG----------------------------

#ifndef __DEBUG__
#define __DEBUG__ true
#endif // !__DEBUG__

#if __DEBUG__ == true
#define COUT_DEBUG(...) std::cout << __VA_ARGS__ << std::endl;
#define LOG_DEBUG(...) printf(__VA_ARGS__);
#define LOG_DEBUG_IF( condition, ...) if(condition) LOG_DEBUG(__VA_ARGS__);
#define DEBUG(...) __VA_ARGS__;
#else
#define COUT_DEBUG(...)
#define LOG_DEBUG(...)
#define LOG_DEBUG_IF(...)
#define DEBUG(...)
#endif

#define LOG_CONFIG(...) printf(__VA_ARGS__)

//------------------------------------------------------------------------


//--------------------------CLASS TEMPLATES-------------------------------


#define RENDER_KERNEL_LAUNCHER_TEMPLATES \
template class StreamingVolPTmk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>; \
template class StreamingVolPTmk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>; \
template class StreamingVolPTsk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>; \
template class StreamingVolPTsk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>; \
template class SortingVolPTsk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>; \
template class SortingVolPTsk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>; \
template class RegenerationVolPTsk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>; \
template class RegenerationVolPTsk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>; \
template class NaiveVolPTmk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>; \
template class NaiveVolPTmk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>; \
template class NaiveVolPTsk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>; \
template class NaiveVolPTsk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>; \

#define CUDAVOLPATH_TEMPLATES \
template class CudaVolPath<StreamingVolPTsk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>>; \
template class CudaVolPath<StreamingVolPTsk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>>; \
template class CudaVolPath<SortingVolPTsk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>>; \
template class CudaVolPath<SortingVolPTsk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>>; \
template class CudaVolPath<StreamingVolPTmk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>>; \
template class CudaVolPath<StreamingVolPTmk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>>; \
template class CudaVolPath<RegenerationVolPTsk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>>; \
template class CudaVolPath<RegenerationVolPTsk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>>; \
template class CudaVolPath<NaiveVolPTmk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>>; \
template class CudaVolPath<NaiveVolPTmk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>>; \
template class CudaVolPath<NaiveVolPTsk<SimpleVolumeDeviceScene<DeviceMedium, GGX>>>; \
template class CudaVolPath<NaiveVolPTsk<SimpleVolumeDeviceScene<HostDeviceMedium, GGX>>>; \
//------------------------------------------------------------------------


#endif /* DEFINES_H_ */
