#ifndef PROJECTIONS_H_
#define PROJECTIONS_H_

#include <cuda_runtime.h>

__device__ void projD(float* d_x, int w, int h);
__device__ void projC(float* d_x, float gamma, int w, int h);

#endif