#ifndef IMAGE_TRANSFORM_H_
#define IMAGE_TRANSFORM_H_

#include <cuda_runtime.h>

/**
 * Halves width and height of the input image by first blurring it,
 * and then copying every 4th pixel to the according position
 * in the output image.
 * x and y for the output (small) image, w, h and nc of the input (big) image
 * w and h are assumed to be multiples of 2
**/
__device__ void downsample(int x, int y, int c, float* in, int w, int h, float* out);

/**
 * To be called with one thread per pixel of the output (big) image!
 * Doubles width and height of the input image,
 * analogously to the downsample function
 * w, h and nc of the input (small) image
**/
__global__ void upsample(float* in, float* out, int w, int h);


__global__ void blur(float *in, float *out, int w, int h, float kernelDia);

__device__ float d_upsample(float* in, int x_big, int y_big, int c, int w_big, int h_big);
__device__ float d_downsample(float* in, int x_small, int y_small, int c, int w_small, int h_small);

float getKernel(float * kernel, float sigma, int diameter);
void getNormalizedKernel(float * kernel, float sigma, int diameter);

// gaussian blur (5x5 kernel for sigma = 1.5) as kernel function
__global__ void gaussBlur5(float* in, float* out, int w, int h);
#endif
