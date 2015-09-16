#ifndef FLOW_COLOR_H_
#define FLOW_COLOR_H_

#include <cuda_runtime.h>

#define PI 3.1415926535f
#define EPSILON 0.0001

__global__ void createColorCoding(float* d_v1, float* d_v2, float* d_out, int w, int h);
__global__ void createColorCoding(float* d_in, float* d_v1, float* d_v2, float* d_out, int w, int h, int nc);
__global__ void createColorCoding(float* d_v1, float* d_v2, float* d_out, int w, int h, int border);

#endif