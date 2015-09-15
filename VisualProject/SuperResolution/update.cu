#include "update.h"

__global__ void updateP(float* d_p, float* d_v1, float* d_v2, float2* d_A, float* d_b, float sigma, float gamma, int w, int h) {
	// get current thread index (x, y, c)
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockDim.y;
	int c = threadIdx.z;

	// return if coordinate (x, y) not inside image
	if (x >= w || y >= h) {
		return;
	}

	// index for access without color channel
	int idx = x + w * y;
	// index for access with color channel
	int idxc = idx + w * h * c;

	// compute:
	// (1) p_k+1' = proj_C(p_k + s * (Av + b))
	// (2) p_k+1 = 2 * p_k+1' - p_k

	// b
	float acc = d_b[idxc];
	// Av + b
	acc += d_A[idxc].x * d_v1[idx] + d_A[idxc].y * d_v2[idx];
	// p + s * (Av + b)
	float oldp = d_p[idxc];
	acc = oldp + sigma * acc;
	// proj_C(p_k + s * (Av + b))
	acc = projC(acc, gamma);
	// sor: p_k+1 = 2 * p_k+1' - p_k
	d_p[idxc] = 2 * acc - oldp;
}

__global__ void updateQ(float2* d_q, float* d_v, float sigma, int w, int h) {
	// get current thread index (x, y)
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockDim.y;

	// return if coordinate (x, y) not inside image
	if (x >= w || y >= h) {
		return;
	}

	// index for access image pixel
	int idx = x + w * y;

	// compute:
	// (1) q_k+1' = proj_D(q_k + s * (dv/dx dv/dy))
	// (2) q_k+1 = 2 * q_k+1' - q_k

	// dv/dx dv/dy
	float2 acc = gradient(d_v, x, y, 0, w, h);
	// q_k + s * (dv/dx dv/dy)
	float2 qold = d_q[idx];
	acc = make_float2(qold.x + sigma * acc.x, qold.y + sigma * acc.y);
	// proj_D(q_k + s * (dv/dx dv/dy))
	acc = projD(acc);
	// sor: q_k+1 = 2 * q_k+1' - q_k
	d_q[idx] = make_float2(2 * acc.x - qold.x, 2 * acc.y - qold.y);
}

__global__ void updateV(float* d_v1, float* d_v2, float* d_p, float2* d_q1, float2* d_q2, float2* d_A, float tau, int w, int h) {
	// get current thread index (x, y)
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockDim.y;

	// return if coordinate (x, y) not inside image
	if (x >= w || y >= h) {
		return;
	}

	// index for access image pixel
	int idx = x + w * y;

	// compute:
	// (1) v_k+1 = v_k - t * (A * p_k+1 - (div(q1_k+1) div(q2_k+1)))

	// div(q1_k+1)
	float div_q1 = divergence(d_q1, x, y, w, h);
	// div(q2_k+1)
	float div_q2 = divergence(d_q2, x, y, w, h);
	// p_k+1
	float p1 = d_p[idx];
	float p2 = d_p[idx + w*h];
	float p3 = d_p[idx + 2 * w*h];
	// A * p_k+1 - (div(q1_k+1) div(q2_k+1))
	float acc1 = d_A[idx].x * p1 + d_A[idx + w*h].x * p2 + d_A[idx + 2 * w*h].x * p3 - div_q1;
	float acc2 = d_A[idx].y * p1 + d_A[idx + w*h].y * p2 + d_A[idx + 2 * w*h].y * p3 - div_q2;
	// v_k - t * (A * p_k+1 - (div(q1_k+1) div(q2_k+1)))
	d_v1[idx] -= tau * acc1;
	d_v2[idx] -= tau * acc2;
}