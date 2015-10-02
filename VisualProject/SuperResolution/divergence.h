#ifndef DIVERGENCE_H_
#define DIVERGENCE_H_

#include <cuda_runtime.h>

// device functions per pixel

// gradient per pixel with central differences and Neumann boundary conditions (set to 0.0) -> returns result as float2
__device__ float2 gradientCD(float* d_in, int x, int y, int c, int w, int h);
// gradient per pixel with forward differences and Neumann boundary conditions (set to 0.0) -> stores x derivative in out 1 and y derivative in out2
__device__ void gradient(float* d_in, float* d_out1, float* d_out2, int x, int y, int c, int w, int h);
// gradient per pixel with forward differences and Neumann boundary conditions (set to 0.0) -> returns result as float2
__device__ float2 gradient(float* d_in, int x, int y, int c, int w, int h);
// divergence per pixel with backward differences and Dirichlet boundary conditions -> stores result in out
__device__ void divergence(float* d_out, float2* d_in, int x, int y, int w, int h, int c = 0);
// divergence per pixel with backward differences and Dirichlet boundary conditions -> rturn result as float
__device__ float divergence(float2* d_q, int x, int y, int w, int h, int c = 0);


// kernel functions per image

// kernel for gradient with forward differences and Neumann boundary conditions (set to 0.0) -> stores x derivative in out1 and y derivative in out2
__global__ void calculateGradient(float* d_in, float* d_out1, float* d_out2, int w, int h, int nc);
// kernel for gradient with forward differences and Neumann boundary conditions (set to 0.0)
__global__ void calculateGradient(float* d_in, float2* d_out, int w, int h, int nc);
// kernel for gradient with central differences and Neumann boundary conditions (set to 0.0)
__global__ void calculateGradientCD(float* d_in, float2* d_out, int w, int h, int nc);

// kernel for difference between two images im2 - im1 -> stores result in diff
__global__ void imageDiff(float* d_im1, float* d_im2, float* d_diff, int w, int h);

#endif
