#ifndef UPDATE_H_
#define UPDATE_H_

#include <cuda_runtime.h>

__global__ void updateP(float* d_p, float* d_v1, float* d_v2, float* d_A1, float* d_A2, float* d_b, float sigma, int w, int h, int nc);
__global__ void updateQ(float* d_q, float* d_v, float sigma, int w, int h, int nc);
__global__ void updateV(float* d_v, float* d_p, float* d_q1, float* d_q2, float* d_A1, float* d_A2, float tau, int w, int h, int nc);

#endif