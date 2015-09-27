#include "updateSuperResolution.h"
#include "imageTransform.h"
#include "projections.h"
#include "divergence.h"
#include "helper_math.h"

__global__ void super_updateP(float* d_p, float* d_f, cudaTextureObject_t* d_u, float sigma, float alpha, int w, int h) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;
	if (x >= w || y >= h) return;

	int idx = x + y * w + c * w * h;

	float pOld = d_p[idx];

	// p + sig*(Au-f)

	//float sampVal = d_downsample(d_u, x, y, c, w, h);
	float normCoordsX = float(x + 0.5f) / w;
	float normCoordsY = float(y + 0.5f) / h;
	float sampVal = tex2D<float>(d_u[c], normCoordsX, normCoordsY);

	float pNew = pOld + sigma * (sampVal - d_f[idx]);
	
	// projC
	pNew = projL1(pNew, alpha);

	d_p[idx] = 2 * pNew - pOld;
}

__global__ void super_updateQ(float2* d_q, float* d_u, float sigma, float beta, int w, int h) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;
	if (x >= w || y >= h) return;

	int idx = x + y * w + c * w * h;

	// gradient of U
	float2 gradU = gradient(d_u, x, y, c, w, h);

	float2 qOld = d_q[idx];

	// q * sigma * gradV
	float2 qNew = qOld + sigma * gradU;
	// projE
	qNew = projL2(qNew, beta);

	d_q[idx] = 2 * qNew - qOld;
}

__global__ void super_updateR(float* d_r, float* d_u1, float* d_u2, float* d_v1, float* d_v2, float gamma, int w, int h) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;
	if (x >= w || y >= h) return;

	int idx = x + y * w;
	int idxc = idx + c * w * h;


	float rOld = d_r[idxc];
	float v1 = d_v1[idx];
	float v2 = d_v2[idx];
	float u1 = d_u1[idxc];
	float u2 = d_u2[idxc];

	// backward difference with Dirichlet boundaries
	//float difUx = (x > 0) ? u2 - d_u2[idxc - 1] : u2;
	//float difUy = (y > 0) ? u2 - d_u2[idxc - w] : u2;

	// forward difference with neumann boundaries
	float difUx = (x < w-1) ? d_u2[idxc + 1] - u2 : 0.0f;
	float difUy = (y < h-1) ? d_u2[idxc + w] - u2 : 0.0f;

	// central difference with neumann boundaries
	//float difUx = (x < w - 1 && x > 0) ? d_u2[idxc + 1] - d_u2[idxc - 1] : 0.0f;
	//float difUy = (y < h - 1 && y > 0) ? d_u2[idxc + w] - d_u2[idxc - w] : 0.0f;

	// calc sigma
	float sigma = 1.0f / (2 + 2 * fabsf(v1) + 2 * fabsf(v2));
	// r + sigma * B (u1,u2)
	float rNew = rOld + sigma * (u1 - u2 - difUx*v1 - difUy*v2);
	// projG
	rNew = projL1(rNew, gamma);
	d_r[idxc] = 2 * rNew - rOld;
}

__global__ void super_updateU(float * d_u1, float * d_u2, float * d_r, cudaTextureObject_t* d_p1, cudaTextureObject_t* d_p2,
	float2 * d_q1, float2 * d_q2, float * d_v1, float * d_v2, float gamma, int w, int h) {
	
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;
	if (x >= w || y >= h) return;

	int idx = x + y * w;
	int idxc = idx + c * w * h;

	float u1Old = d_u1[idxc];
	float u2Old = d_u2[idxc];
	float r = d_r[idxc];
	float v1 = d_v1[idx];
	float v2 = d_v2[idx];
	
	// backward difference with Dirichlet boundaries
	float difRx = (x > 0) ? r - d_r[idxc - 1] : r;
	float difRy = (y > 0) ? r - d_r[idxc - w] : r;

	// forward difference with Neumann boundaries
	//float difRx = (x < w - 1) ? d_r[idxc + 1] - r : 0.0f;
	//float difRy = (y < h - 1) ? d_r[idxc + w] - r : 0.0f;

	// central difference with Neumann boundaries
	//float difRx = (x < w - 1 && x > 0) ? d_r[idxc + 1] - d_r[idxc - 1] : 0.0f;
	//float difRy = (y < h - 1 && y > 0) ? d_r[idxc + w] - d_r[idxc - w] : 0.0f;

	// calc s with the differences
	float s1 = r;
	float s2 = -r + v1 * difRx + v2 * difRy;
	
	// div(q)
	float divQ1 = divergence(d_q1, x, y, w, h, c);
	float divQ2 = divergence(d_q2, x, y, w, h, c);

	// tau
	float t1 = 1.0f / 6.0f;
	float t2 = 1.0f / (6.0f + 2 * fabsf(v1) + 2 * fabsf(v2));

	// A^T*p

	//float sampVal1 = d_upsample(d_p1, x, y, c, w, h);
	//float sampVal2 = d_upsample(d_p2, x, y, c, w, h);

	float normCoordsX = float(x + 0.5f) / w;
	float normCoordsY = float(y + 0.5f) / h;
	float sampVal1 = tex2D<float>(d_p1[c], normCoordsX, normCoordsY);
	float sampVal2 = tex2D<float>(d_p2[c], normCoordsX, normCoordsY);

	// update step
	d_u1[idxc] = u1Old - t1 * (sampVal1 - divQ1 + s1);
	d_u2[idxc] = u2Old - t2 * (sampVal2 - divQ2 + s2);


}