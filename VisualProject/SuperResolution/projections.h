#ifndef PROJECTIONS_H_
#define PROJECTIONS_H_

#include <cuda_runtime.h>

/**
 * Normale vector x if it is longer than the given limit
 */
__device__ float2 projL2(float2 x);

/**
 * Clamp x to maximum/minimum value of the given limit
 */
__device__ float projL1(float x, float limit);

#endif
