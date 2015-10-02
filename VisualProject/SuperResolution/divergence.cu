#include "divergence.h"


// kernel for gradient with central differences and Neumann boundary conditions (set to 0.0)
__global__ void calculateGradientCD(float* d_in, float2* d_out, int w, int h, int nc) {
	// get current thread index (x, y, c)
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int c = threadIdx.z;

	// return if coordinate (x, y) not inside image
	if (x >= w || y >= h) {
		return;
	}

	// index for image pixel access
	int ind = x + y * w + c * w * h;

	// compute gradient and store in output image
	d_out[ind] = gradientCD(d_in, x, y, c, w, h);
}


// gradient per pixel with central differences and Neumann boundary conditions (set to 0.0) -> returns result as float2
__device__ float2 gradientCD(float* d_in, int x, int y, int c, int w, int h) {
	// index for image pixel access
	int ind = x + y * w + c * w * h;

	// define result
	float2 result = { 0.0f, 0.0f };
	if (x < w - 1 && x > 0) {
		result.x = (d_in[ind + 1] - d_in[ind - 1]) / 2.0f;
	}
	if (y < h - 1 && y > 0) {
		result.y = (d_in[ind + w] - d_in[ind - w]) / 2.0f;
	}
	return result;
}


// kernel for gradient with forward differences and Neumann boundary conditions (set to 0.0)
__global__ void calculateGradient(float* d_in, float2* d_out, int w, int h, int nc) {
	// get current thread index (x, y, c)
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int c = threadIdx.z;

	// return if coordinate (x, y) not inside image
	if (x >= w || y >= h) {
		return;
	}

	// index for image pixel access
	int ind = x + y*w + c*w*h;

	// compute gradient and store in output image
	d_out[ind] = gradient(d_in, x, y, c, w, h);
}


// kernel for gradient with forward differences and Neumann boundary conditions (set to 0.0) -> stores x derivative in out1 and y derivative in out2
__global__ void calculateGradient(float* d_in, float* d_out1, float* d_out2, int w, int h, int nc) {
	// get current thread index (x, y, c)
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int c = threadIdx.z;

	// return if coordinate (x, y) not inside image
	if (x >= w || y >= h) {
		return;
	}

	// compute gradient and store result in out1, out2 implicitly
	gradient(d_in, d_out1, d_out2, x, y, c, w, h);
}


// gradient per pixel with forward differences and Neumann boundary conditions (set to 0.0) -> stores x derivative in out 1 and y derivative in out2
__device__ void gradient(float* d_in, float* d_out1, float* d_out2, int x, int y, int c, int w, int h) {
	// index for image pixel access
	int ind = x + y*w + c*w*h;

	// compute gradient
	float2 grad = gradient(d_in, x, y, c, w, h);

	// store result in output images
	d_out1[ind] = grad.x;
	d_out2[ind] = grad.y;
}


// gradient per pixel with forward differences and Neumann boundary conditions (set to 0.0) -> returns result as float2
__device__ float2 gradient(float* d_in, int x, int y, int c, int w, int h) {
	// index for image pixel access
	int ind = x + y*w + c*w*h;

	// define result
	float2 result = { 0.0f, 0.0f };
	if (x < w - 1) {
		result.x = d_in[ind + 1] - d_in[ind];
	}
	if (y < h - 1) {
		result.y = d_in[ind + w] - d_in[ind];
	}
	return result;
}


// divergence per pixel with backward differences and Dirichlet boundary conditions -> stores result in out
__device__ void divergence(float* d_out, float2* d_in, int x, int y, int w, int h, int c) {
	// index for image pixel access
	int ind = x + y*w + w*h*c;

	// store result in output image
	d_out[ind] = divergence(d_in, x, y, w, h, c);
}

// divergence per pixel with backward differences and Dirichlet boundary conditions -> rturn result as float
__device__ float divergence(float2* d_in, int x, int y, int w, int h, int c) {
	// index for image pixel access
	int ind = x + y*w + w*h*c;

	// define result
	float2 inVal = d_in[ind];
	float ret = inVal.x + inVal.y;
	if (x > 0) {
		ret -=  d_in[ind - 1].x;
	}
	if (y > 0) {
		ret -=  d_in[ind - w].y;
	}
	return ret;
}


// Computes the difference between two images in all color channels: d_diff = d_im2 - d_im1
__global__ void imageDiff(float* d_im1, float* d_im2, float* d_diff, int w, int h) {
	// get current thread index (x, y, c)
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;

	// return if coordinate (x, y) not inside image
	if (x >= w || y >= h) {
		return;
	}

	// index for image pixel access
	int idx = x + y * w + c * w * h;

	// store result in output image
	d_diff[idx] = d_im2[idx] - d_im1[idx];
}