#include "updateSuperResolution.h"
#include "imageTransform.h"
#include "projections.h"
#include "divergence.h"
#include "helper_math.h"

__global__ void super_updateP(float* d_p, float* d_f, float* d_Au, float sigma, float alpha, int w, int h) {
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

	// previous value of p
	float pOld = d_p[idx];

	// p + sigma * (Au - f)
	float pNew = pOld + sigma * (d_Au[idx] - d_f[idx]);
	
	// projL1 to alpha
	pNew = projL1(pNew, alpha);

	// update p
	d_p[idx] = 2 * pNew - pOld;
}

__global__ void super_updateQ(float2* d_q, float* d_u, float sigma, float beta, int w, int h) {
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

	// gradient of u
	float2 gradU = gradient(d_u, x, y, c, w, h);

	// previous value of q
	float2 qOld = d_q[idx];

	// q + sigma * gradient(u)
	float2 qNew = qOld + sigma * gradU;

	// projL2 to beta
	qNew = projL2(qNew, beta);

	// update q
	d_q[idx] = 2 * qNew - qOld;
}

__device__ float applyBflow(float* d_u1, float* d_u2, float v1, float v2, int x, int y, int c, int w, int h) {
	// index for image pixel access without color channel
	int idx = x + y * w;
	// index for image pixel access with color channel
	int idxc = idx + c * w * h;

	// backward difference with Dirichlet boundaries
	//float difU2x = (x > 0) ? u2 - d_u2[idxc - 1] : u2;
	//float difU2y = (y > 0) ? u2 - d_u2[idxc - w] : u2;

	// forward difference with neumann boundaries
	//float difU2x = (x < w-1) ? d_u2[idxc + 1] - u2 : 0.0f;
	//float difU2y = (y < h-1) ? d_u2[idxc + w] - u2 : 0.0f;

	// central difference with Neumann boundaries -> set to 0.0
	float difU2x, difU2y;
	if (x < w - 1 && x > 0) {
		difU2x = (d_u2[idxc + 1] - d_u2[idxc - 1]) / 2;
	} else {
		difU2x = 0.0f;
	}
	if (y < h - 1 && y > 0) {
		difU2y = (d_u2[idxc + w] - d_u2[idxc - w]) / 2;
	} else {
		difU2y = 0.0f;
	}

	// r + sigma * B (u1,u2)^T
	return (d_u1[idxc] - d_u2[idxc] - v1 * difU2x - v2 * difU2y);
}

__global__ void super_updateR(float* d_r, float* d_u1, float* d_u2, float* d_v1, float* d_v2, float gamma, int w, int h) {
	// get current thread index (x, y, c)
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;

	// return if coordinate (x, y) not inside image
	if (x >= w || y >= h) {
		return;
	}

	// index for image pixel access without color channel
	int idx = x + y * w;
	// index for image pixel access with color channel
	int idxc = idx + c * w * h;

	// previous r, v1, v2
	float rOld = d_r[idxc];
	float v1 = d_v1[idx];
	float v2 = d_v2[idx];

	// compute step size sigma
	float sigma = 1.0f / (2 + 2 * fabsf(v1) + 2 * fabsf(v2));

	// apply operator B_flow
	float rNew = rOld + sigma * applyBflow(d_u1, d_u2, v1, v2, x, y, c, w, h);

	// projL1 to gamma
	rNew = projL1(rNew, gamma);

	// update r
	d_r[idxc] = 2 * rNew - rOld;
}

__device__ float2 applyBflowTranspose(float* d_r, float v1, float v2, int x, int y, int c, int w, int h) {
	// index for image pixel access without color channel
	int idx = x + y * w;
	// index for image pixel access with color channel
	int idxc = idx + c * w * h;

	// backward difference with Dirichlet boundaries
	//float difRx = (x > 0) ? r - d_r[idxc - 1] : r;
	//float difRy = (y > 0) ? r - d_r[idxc - w] : r;

	// forward difference with Neumann boundaries
	//float difRx = (x < w - 1) ? d_r[idxc + 1] - r : 0.0f;
	//float difRy = (y < h - 1) ? d_r[idxc + w] - r : 0.0f;

	// transpose operator of the central differences (-> with two pixel Dirichlet boundaries)
	float difRx, difRy;
	if (x <= 1) {
		difRx = -0.5f * d_r[idxc + 1];
	} else if (x >= w - 2) {
		difRx = 0.5f * d_r[idxc - 1];
	} else {
		difRx = (d_r[idxc - 1] - d_r[idxc + 1]) / 2;
	}
	if (y <= 1) {
		difRy = -0.5f * d_r[idxc + w];
	} else if (y >= h - 2) {
		difRy = 0.5f * d_r[idxc - w];
	} else {
		difRy = (d_r[idxc - w] - d_r[idxc + w]) / 2;
	}

	// compute s with the differences
	float s1 = d_r[idxc];
	float s2 = -d_r[idxc] - v1 * difRx - v2 * difRy;

	return make_float2(s1, s2);
}

__global__ void super_updateU(float * d_u1, float * d_u2, float * d_r, float* d_Atp1, float* d_Atp2,
	float2 * d_q1, float2 * d_q2, float * d_v1, float * d_v2, int w, int h) {
	// get current thread index (x, y, c)
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;

	// return if coordinate (x, y) not inside image
	if (x >= w || y >= h) {
		return;
	}

	// index for image pixel access without color channel
	int idx = x + y * w;
	// index for image pixel access with color channel
	int idxc = idx + c * w * h;

	// previous values of u1, u2, r, v1, v2
	float u1Old = d_u1[idxc];
	float u2Old = d_u2[idxc];
	float v1 = d_v1[idx];
	float v2 = d_v2[idx];
	
	// s = B_flow^T r
	float2 s = applyBflowTranspose(d_r, v1, v2, x, y, c, w, h);

	// div(q)
	float divQ1 = divergence(d_q1, x, y, w, h, c);
	float divQ2 = divergence(d_q2, x, y, w, h, c);

	// compute step size tau
	float t1 = 1.0f / 6.0f;
	float t2 = 1.0f / (6.0f + 2 * fabsf(v1) + 2 * fabsf(v2));

	// A^T*p
	//float sampVal1 = d_upsample(d_p1, x, y, c, w, h);
	//float sampVal2 = d_upsample(d_p2, x, y, c, w, h);

	// update step
	d_u2[idxc] = u2Old - t2 * (d_Atp2[idxc] - divQ2 + s.y);
	d_u1[idxc] = u1Old - t1 * (d_Atp1[idxc] - divQ1 + s.x);
}

__global__ void checkB(float* out, float* d_r, float* d_u1, float* d_u2, float* d_v1, float* d_v2, int w, int h) {
	// get current thread index (x, y, c)
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;

	// return if coordinate (x, y) not inside image
	if (x >= w || y >= h) {
		return;
	}

	// index for image pixel access without color channel
	int idx = x + y * w;
	// index for image pixel access with color channel
	int idxc = idx + c * w * h;

	// previous r, u1, u2, v1, v2
	float rOld = d_r[idxc];
	float u1Old = d_u1[idxc];
	float u2Old = d_u2[idxc];
	float v1 = d_v1[idx];
	float v2 = d_v2[idx];

	// apply operator B_flow
	float rDual = applyBflow(d_u1, d_u2, v1, v2, x, y, c, w, h);

	// apply operator B_flow^T
	float2 uDual = applyBflowTranspose(d_r, v1, v2, x, y, c, w, h);

	// write absolute difference into result image -> should be 0.0 if correct
	out[idxc] = fabsf(rDual * d_r[idxc] - uDual.x * d_u1[idxc] + uDual.y * d_u2[idxc]);
}