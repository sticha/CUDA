#ifndef ENERGY_H_
#define ENERGY_H_

#include <cuda_runtime.h>

__global__ void flowFieldEnergy(float* result, float2* d_A, float* d_b, float* d_v1, float* d_v2, float gamma, int w, int h, int nc);
__global__ void superResolutionEnergy(float* result, float* d_u1, float* d_u2, float* d_f1, float* d_f2, float* d_v1, float* d_v2, float alpha, float beta, float gamma, int w, int h, int nc);

#endif