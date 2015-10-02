#ifndef FLOW_COLOR_H_
#define FLOW_COLOR_H_

#include <cuda_runtime.h>

// compute a color coding for the given flow field adding a colored border indicating the direction
__global__ void createColorCoding(float* d_v1, float* d_v2, float* d_out, int w, int h, int border);
// compute a color coding for the given flow field adding a colored border indicating the direction + blend with the input image by alpha = 0.5
__global__ void createColorCoding(float* d_in, float* d_v1, float* d_v2, float* d_out, int w, int h, int nc, int border);

#endif