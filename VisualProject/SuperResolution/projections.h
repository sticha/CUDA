#ifndef PROJECTIONS_H_
#define PROJECTIONS_H_

#include <cuda_runtime.h>

/**
 * Normale vector x if it is longer than 1
 */
__device__ float2 projD(float2 x);
__device__ float2 projL2(float2 x, float limit);

/**
 * Clamp x to maximum value of gamma
 */
__device__ float projC(float x, float gamma);
__device__ float projL1(float x, float limit);
#endif

