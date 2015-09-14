#ifndef DIVERGENCE_H_
#define DIVERGENCE_H_

#include <cuda_runtime.h>

__device__ void gradient(float* d_u, float* d_v1, float* d_v2);
__device__ void divergence(float* d_div, float* d_v1, float* d_v2);

#endif