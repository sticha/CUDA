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

#endif
