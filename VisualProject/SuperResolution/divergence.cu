#include "divergence.h"

__global__ void calculateGradientCD(float* d_u, float2* d_v, int w, int h, int nc) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int c = threadIdx.z;

	if (x >= w || y >= h || c >= nc) return;
	int ind = x + y*w + c*w*h;

	float2 grad = gradientCD(d_u, x, y, c, w, h);
	d_v[ind] = grad;
}

__device__ float2 gradientCD(float* d_u, int x, int y, int c, int w, int h) {
	int ind = x + y*w + c*w*h;
	float2 ret;

	if (x == w - 1 || x == 0) {
		ret.x = 0.f;
	} else {
		ret.x = (d_u[ind + 1] - d_u[ind - 1]) / 2.0f;
	}

	if (y == h - 1 || y == 0) {
		ret.y = 0.f;
	} else {
		ret.y = (d_u[ind + w] - d_u[ind - w]) / 2.0f;
	}

	return ret;
}

__global__ void calculateGradient(float* d_u, float2* d_v, int w, int h, int nc) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int c = threadIdx.z;

	if (x >= w || y >= h || c >= nc) return;
	int ind = x + y*w + c*w*h;

	float2 grad = gradient(d_u, x, y, c, w, h);
	d_v[ind] = grad;
}

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
	} else {
		ret.x = d_u[ind + 1] - d_u[ind];
	}

	if (y == h - 1) {
		ret.y = 0.f;
	} else {
		ret.y = d_u[ind + w] - d_u[ind];
	}

	return ret;
}

__device__ void divergence(float* d_div, float2* d_q, int x, int y, int w, int h, int c) {
	int ind = x + y*w + w*h*c;
	d_div[ind] = divergence(d_q, x, y, w, h, c);
}

__device__ float divergence(float2* d_q, int x, int y, int w, int h, int c) {
	int ind = x + y*w + w*h*c;
	float ret = 0.f;

	if (x > 0) {
		ret += d_q[ind].x - d_q[ind - 1].x;
	} else {
		ret += d_q[ind].x;
	}
	if (y > 0) {
		ret += d_q[ind].y - d_q[ind - w].y;
	} else {
		ret += d_q[ind].y;
	}

	return ret;
}

// Computes the difference between two images in all color channels: d_diff = d_im2 - d_im1
__global__ void imageDiff(float* d_im1, float* d_im2, float* d_diff, int w, int h) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;

	if (x >= w || y >= h)
		return;

	int idx = x + y * w + c * w * h;
	d_diff[idx] = d_im2[idx] - d_im1[idx];
}