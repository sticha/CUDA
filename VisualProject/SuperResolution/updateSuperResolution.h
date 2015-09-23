#ifndef UPDATE_SUPER_RESOLUTION_H_
#define UPDATE_SUPER_RESOLUTION_H_

#include <cuda_runtime.h>

__global__ void super_updateP(float * d_p, float * d_f, float sigma, float alpha, int w, int h);
__global__ void super_updateQ(float2 * d_q, float * d_u, float sigma, float beta, int w, int h);
__global__ void super_updateR(float * d_r, float * d_u1, float * d_u2, float * d_v1, float * d_v2, float gamma, int w, int h);
__global__ void super_updateU(float * d_u1, float * d_u2, float * d_r, float * d_p1, float * d_p2, float2 * d_q1, float2 * d_q2, float * d_v1, float * d_v2, float gamma, int w, int h);

#endif