#ifndef UPDATE_H_
#define UPDATE_H_

#include <cuda_runtime.h>

__global__ void updateP(float* d_p, float* d_v1, float* d_v2, float2* d_A, float* d_b, float gamma, int w, int h);
__global__ void updateQ(float2* d_q, float* d_v, float sigma, int w, int h);
__global__ void updateV(float* d_v1, float* d_v2, float* d_p, float2* d_q1, float2* d_q2, float2* d_A, int w, int h, int nc);

#endif
