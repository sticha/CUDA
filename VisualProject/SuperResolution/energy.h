#ifndef ENERGY_H_
#define ENERGY_H_

#include <cuda_runtime.h>

#define EPSILON 0.0001

__global__ void flowFieldEnergy(float* d_A, float* d_b, float* d_v1, float* d_v2, int w, int h, int nc);

#endif