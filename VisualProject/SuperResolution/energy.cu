#include "energy.h"
#include "divergence.h"
#include <math.h>

__device__ float length(float2 vec) {
	return sqrtf(vec.x*vec.x + vec.y*vec.y);
}

__global__ void flowFieldEnergy(float* result, float2* d_A, float* d_b, float* d_v1, float* d_v2, float gamma, int w, int h, int nc) {
	// shared memory for summation process
	extern __shared__ float sdata[];

	// get current thread index (global and local)
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int tx = threadIdx.x;
	int n = w * h;

	// initialize energy (stays 0.0 for all threads outside image)
	float energy = 0.0f;

	// get energy value only if coordinate index inside image
	if (x < n) {
		// compute energy per pixel with the formula
		// E = g * |b + Av|_1 + |grad(v1)|_2 + |grad(v2)|_2

		// |b + Av|_1
		for (int c = 0; c < nc; c++) {
			// A
			float2 A = d_A[x + n*c];
			energy += abs(d_b[x + c*n] + A.x * d_v1[x] + A.y * d_v2[x]);
		}
		// grad(vi)
		float2 grad_v1 = gradient(d_v1, x % w, x / w, 0, w, h);
		float2 grad_v2 = gradient(d_v2, x % w, x / w, 0, w, h);
		// g * |b + Av|_1 + |grad(v1)|_2 + |grad(v2)|_2
		energy = gamma * energy + length(grad_v1) + length(grad_v2);
	}

	// fill shared memory for summation
	sdata[tx] = energy;
	__syncthreads();

	// sum up energy values
	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
		if (tx < offset) {
			sdata[tx] += sdata[tx + offset];
		}
		__syncthreads();
	}

	// add sum to total sum as result
	if (tx == 0) {
		atomicAdd(result, sdata[tx]);
	}
}

__global__ void superResolutionEnergy(float* result, float* d_u1, float* d_u2, float* d_f1, float* d_f2, float* d_v1, float* d_v2, float alpha, float beta, float gamma, int w, int h, int nc) {
	// TODO
}