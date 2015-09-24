#ifndef DIVERGENCE_H_
#define DIVERGENCE_H_

#include <cuda_runtime.h>

__device__ float2 gradientCD(float* d_u, int x, int y, int c, int w, int h);
__device__ void gradient(float* d_u, float* d_v1, float* d_v2, int x, int y, int c, int w, int h);
__device__ float2 gradient(float* d_u, int x, int y, int c, int w, int h);
__device__ void divergence(float* d_div, float2* d_q, int x, int y, int w, int h, int c = 0);
__device__ float divergence(float2* d_q, int x, int y, int w, int h, int c = 0);

__global__ void imageDiff(float* d_im1, float* d_im2, float* d_diff, int w, int h);

/**
 * CUDA Kernel, that calculates all 
 */
__global__ void calculateGradient(float* d_u, float* d_v1, float* d_v2, int w, int h, int nc);
__global__ void calculateGradient(float* d_u, float2* d_v, int w, int h, int nc);
__global__ void calculateGradientCD(float* d_u, float2* d_v, int w, int h, int nc);

#endif
