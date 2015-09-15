#include "divergence.h"

__device__ void gradient(float* d_u, float* d_v1, float* d_v2, int x, int y, int c, int w, int h){
	int ind = x + y*w + c*w*h;

	if (x == w - 1) {
		d_v1[ind] = 0.f;
	} else {
		d_v1[ind] = d_u[ind + 1] - d_u[ind];
	}

	if (y == h - 1) {
		d_v2[ind] = 0.f;
	}
	else {
		d_v2[ind] = d_u[ind + w] - d_u[ind];
	}
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
