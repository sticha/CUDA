#include "updateSuperResolution.h"
#include "imageTransform.h"
#include "projections.h"
#include "divergence.h"
#include "helper_math.h"

__global__ void super_updateP(float * d_p, float * d_f, float sigma, float alpha, int w, int h) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;

	int idx = x + y * w * h + c * w * h;

	float pOld = d_p[idx];

	// p + sig*(Au-f)
//TODO
	float sampVal;
//	float sampVal = downsample(d_u, x, y);
	float pNew = pOld + sigma * ( sampVal - d_f[idx]);
	
	// projC
	pNew = projL1(pNew, alpha);

	d_p[idx] = 2 * pNew - pOld;
}

__global__ void super_updateQ(float2 * d_q, float * d_u, float sigma, float beta, int w, int h) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;

	int idx = x + y * w * h + c * w * h;

	// gradient of U
	float2 gradU = gradient(d_u, x, y, c, w, h);

	float2 qOld = d_q[idx];

	// q * sigma * gradV
	float2 qNew = qOld + sigma * gradU;
	// projE
	qNew = projL2(qNew, beta);

	d_q[idx] = 2 * qNew - qOld;
}

__global__ void super_updateR(float * d_r, float * d_u1, float * d_u2, float * d_v1, float * d_v2, float gamma, int w, int h) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;

	int idx = x + y * w * h;
	int idxc = idx + c * w * h;


	float rOld = d_r[idxc];
	float v1 = d_v1[idx];
	float v2 = d_v2[idx];
	float u1 = d_u1[idxc];
	float u2 = d_u2[idxc];

	// bw difference with neumann boundaries
	float difUx = (x > 0) ? u2 - d_u2[idxc - 1] : u2;
	float difUy = (y > 0) ? u2 - d_u2[idxc - w] : u2;

	// calc sigma
	float sigma = 1.0f / (2 + 2 * (fabsf(v1), fabsf(v2)));
	// r + sigma * B (u1,u2)
	float rNew = rOld + sigma* (u1 - u2 - difUx*v1 - difUy*v2);
	// projG
	rNew = projL1(rNew, gamma);
	d_r[idxc] = 2 * rNew - rOld;
}

__global__ void super_updateU(float * d_u1, float * d_u2, float * d_r, float * d_p1, float * d_p2, 
	float2 * d_q1, float2 * d_q2, float * d_v1, float * d_v2, float gamma, int w, int h) {
	
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;

	int idx = x + y * w * h;
	int idxc = idx + c * w * h;

	float u1Old = d_u1[idxc];
	float u2Old = d_u2[idxc];
	float r = d_r[idxc];
	float v1 = d_v1[idx];
	float v2 = d_v2[idx];
	
	// bw difference with Neumann boundaries
	float difRx = (x > 0) ? r - d_r[idxc - 1] : r;
	float difRy = (y > 0) ? r - d_r[idxc - w] : r;

	// calc s with the differences
	float s1 = r;
	float s2 = -r + v1 * difRx + v2 * difRy;
	
	// div(q)
	float divQ1 = divergence(d_q1, x, y, c, w, h);
	float divQ2 = divergence(d_q2, x, y, c, w, h);

	// tau
	float t1 = 1.0f / 6.0f;
	float t2 = 1.0f / (6.0f + 2 * v1 + 2 * v2);

	// A^T*p
//TODO
	float sampVal1;
	float sampVal2;
//	float sampVal1 = upsample(d_p1, x, y);
//	float sampVal2 = upsample(d_p2, x, y);

	// update step
	d_u1[idxc] = u1Old - t1 * (sampVal1 - divQ1 + s1);
	d_u2[idxc] = u2Old - t2 * (sampVal2 - divQ2 + s2);


}