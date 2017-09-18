/*
 * Debug.h
 *
 *  Created on: 01/set/2017
 *      Author: macbook
 */

#ifndef DEBUG_H_
#define DEBUG_H_

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cstdio>

#define FILENAME \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define CHECK_CUDA_ERROR(msg) checkCudaErrorFunc(msg, FILENAME, __LINE__)

inline void checkCudaErrorFunc(const char *msg, const char *file, int line) {
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess == err) {
    return;
  }

  fprintf(stderr, "CUDA error");
  if (file != nullptr) {
    fprintf(stderr, " (%s:%d)", file, line);
  }
  fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
  getchar();
#endif
  exit(EXIT_FAILURE);
}

#endif /* DEBUG_H_ */
