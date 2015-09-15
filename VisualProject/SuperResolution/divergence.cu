#include "divergence.h"

__global__ void calculateGradient(float* d_u, float* d_v1, float* d_v2, int w, int h, int nc) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int c = threadIdx.z;

	if (x >= w || y >= h || c >= nc) return;

	gradient(d_u, d_v1, d_v2, x, y, c, w, h);
}

__device__ void gradient(float* d_u, float* d_v1, float* d_v2, int x, int y, int c, int w, int h){
	int ind = x + y*w + c*w*h;

	float2 v = gradient(d_u, x, y, c, w, h);
	d_v1[ind] = v.x;
	d_v2[ind] = v.y;
}

__device__ float2 gradient(float* d_u, int x, int y, int c, int w, int h) {
	int ind = x + y*w + c*w*h;
	float2 ret;

	if (x == w - 1) {
		ret.x = 0.f;
	}
	else {
		ret.x = d_u[ind + 1] - d_u[ind];
	}

	if (y == h - 1) {
		ret.y = 0.f;
	}
	else {
		ret.y = d_u[ind + w] - d_u[ind];
	}

	return ret;
}

__device__ void divergence(float* d_div, float2* d_q, int x, int y, int w, int h) {
	int ind = x + y*w;
	d_div[ind] = divergence(d_q, x, y, w, h);
}

__device__ float divergence(float2* d_q, int x, int y, int w, int h) {
	int ind = x + y*w;
	float ret = 0.f;

	if (x > 0) {
		ret += d_q[ind].x - d_q[ind - 1].x;
	}
	if (y > 0) {
		ret += d_q[ind].y - d_q[ind - w].y;
	}

	return ret;
}
