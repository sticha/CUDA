#ifndef UPDATE_SUPER_RESOLUTION_H_
#define UPDATE_SUPER_RESOLUTION_H_

#include <cuda_runtime.h>

__global__ void super_updateP();
__global__ void super_updateQ();
__global__ void super_updateR();
__global__ void super_updateU();

#endif