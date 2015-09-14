#ifndef PROJECTIONS_H_
#define PROJECTIONS_H_

#include <cuda_runtime.h>

__device__ float2 projD(float2 x);
__device__ float projC(float x, float gamma);

#endif