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
	// index for image pixel access with color channel
	int idxc = x + y * w + c * w * h;

	// forward difference with Neumann boundaries -> set to 0.0
	/*float difU2x, difU2y;
	if (x < w - 1) {
		difU2x = d_u2[idxc + 1] - d_u2[idxc];
	} else {
		difU2x = 0.0f;
	}
	if (y < h - 1) {
		difU2y = d_u2[idxc + w] - d_u2[idxc];
	} else {
		difU2y = 0.0f;
	}*/

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

	// B (u1,u2)^T
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

__device__ float applyBflowTranspose(float* d_r1, float* d_r2, float* d_v1, float* d_v2, int x, int y, int c, int w, int h, int imIndx, int nImgs) {
	// index for image pixel access without color channel
	int idx = x + y * w;
	// index for image pixel access with color channel
	int idxc = idx + c * w * h;

	if (imIndx == 0) {
		// first image
		return d_r2[idxc];
	}

	// transpose operator of the forward differences combined with v
	/*float difRx, difRy;
	if (x == 0) {
		difRx = -d_r1[idxc] * d_v1[idx];
	} else if (x == w - 1) {
		difRx = d_r1[idxc - 1] * d_v1[idx - 1];
	} else {
		difRx = d_r1[idxc - 1] * d_v1[idx - 1] - d_r1[idxc] * d_v1[idx];
	}
	if (y == 0) {
		difRy = -d_r1[idxc] * d_v2[idx];
	} else if (y == h - 1) {
		difRy = d_r1[idxc - w] * d_v2[idx - w];
	} else {
		difRy = d_r1[idxc - w] * d_v2[idx - w] - d_r1[idxc] * d_v2[idx];
	}*/

	// transpose operator of the central differences combined with v (-> with two pixel Dirichlet boundaries)
	float difRx, difRy;
	if (x <= 1) {
		difRx = -0.5f * d_r1[idxc + 1] * d_v1[idx + 1];
	} else if (x >= w - 2) {
		difRx = 0.5f * d_r1[idxc - 1] * d_v1[idx - 1];
	} else {
		difRx = (d_r1[idxc - 1] * d_v1[idx - 1] - d_r1[idxc + 1] * d_v1[idx + 1]) / 2;
	}
	if (y <= 1) {
		difRy = -0.5f * d_r1[idxc + w] * d_v2[idx + w];
	} else if (y >= h - 2) {
		difRy = 0.5f * d_r1[idxc - w] * d_v2[idx - w];
	} else {
		difRy = (d_r1[idxc - w] * d_v2[idx - w] - d_r1[idxc + w] * d_v2[idx + w]) / 2;
	}

	if (imIndx == nImgs - 1) {
		// last image
		return -d_r1[idxc] - difRx - difRy;
	}

	// intermediate images
	return d_r2[idxc] - d_r1[idxc] - difRx - difRy;
}

__global__ void super_updateU(float* d_u, float* d_r1, float* d_r2, float* d_Atp,
	float2* d_q, float* d_v1, float* d_v2, int w, int h, int imIndx, int nImgs) {
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
	
	// s = B_flow^T r
	float s = applyBflowTranspose(d_r1, d_r2, d_v1, d_v2, x, y, c, w, h, imIndx, nImgs);

	// div(q)
	float divQ = divergence(d_q, x, y, w, h, c);

	// compute step size tau
	float tau;
	if (imIndx == 0) {
		tau = 1.0f / 6.0f;
	} else if (imIndx == nImgs - 1) {
		tau = 1.0f / (6.0f + 2 * fabsf(d_v1[idx]) + 2 * fabsf(d_v2[idx]));
	} else {
		tau = 1.0f / (7.0f + 2 * fabsf(d_v1[idx]) + 2 * fabsf(d_v2[idx]));
	}

	// update step
	d_u[idxc] = d_u[idxc] - tau * (d_Atp[idxc] - divQ + s);
}