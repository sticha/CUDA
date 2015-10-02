#ifndef UPDATE_FLOW_FIELD_H_
#define UPDATE_FLOW_FIELD_H_

#include <cuda_runtime.h>

// Update dual variables p for flow field optimization
__global__ void flow_updateP(float* d_p, float* d_v1, float* d_v2, float2* d_A, float* d_b, float gamma, int w, int h);
// Update dual variables q1 or q2 for flow field optimization
__global__ void flow_updateQ(float2* d_q, float* d_v, float sigma, int w, int h);
// Update flow field using dual variables
__global__ void flow_updateV(float* d_v1, float* d_v2, float* d_p, float2* d_q1, float2* d_q2, float2* d_A, int w, int h, int nc);

#endif
