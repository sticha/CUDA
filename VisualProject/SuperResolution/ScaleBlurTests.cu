#include "helper.h"
#include "imageTransform.h"
#include <iostream>

using namespace std;

__global__ void initArrayFixValues(float* array, int w, int h) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;

	if (x >= w || y >= h) return;

	int ind = x + y*w + c*w*h;
	array[ind] = ind;
}

/**__global__ void d_scalarProduct(float* a, float* b, float* blockRes, int w, int h) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;
	int ind = x + w*y + w*h*c;
	int ind_out = threadIdx.x + threadIdx.y*blockDim.x + blockDim.x*blockDim.y*c;

	if (x >= w || y >= h) return;

	float result = a[ind] * b[ind];
	atomicAdd(&blockRes[ind_out], result);
}**/

void execA_U(dim3 grid3d, dim3 grid3d_small, dim3 block3d, float* u, float* Au, int w_small, int w_big, int h_small, int h_big, int nc){
	int smBytes = (block3d.x + 4) * (block3d.y + 4) * sizeof(float);
	float* temp_big;
	cudaMalloc(&temp_big, w_big*h_big*nc*sizeof(float));

	gaussBlur5<<<grid3d, block3d, smBytes>>>(u, temp_big, w_big, h_big);
	cudaDeviceSynchronize();
	CUDA_CHECK;
	downsample<<<grid3d_small, block3d>>>(temp_big, Au, w_big, h_big, w_small, h_small);
	cudaDeviceSynchronize();
	CUDA_CHECK;

	cudaFree(temp_big);
	CUDA_CHECK;
}

void execAt_v(dim3 grid3d, dim3 block3d, float* v, float* Atv, int w_small, int w_big, int h_small, int h_big, int nc) {
	int smBytes = (block3d.x + 4) * (block3d.y + 4) * sizeof(float);
	float* temp_big;
	cudaMalloc(&temp_big, w_big*h_big*nc*sizeof(float));

	// Upsample p1
	upsample<<<grid3d, block3d>>>(v, temp_big, w_big, h_big, w_small, h_small);
	cudaDeviceSynchronize();
	CUDA_CHECK;
	// Blur upsampled p1
	gaussBlur5<<<grid3d, block3d, smBytes>>>(temp_big, Atv, w_big, h_big);
	cudaDeviceSynchronize();
	CUDA_CHECK;

	cudaFree(temp_big);
	CUDA_CHECK;
}

long float scalarProduct(dim3 block3d, dim3 grid3d, float* a, float* b, int w, int h, int nc) {
	//float* blockResult;
	//float* blockResult_local;
	//int blockResultSize = block3d.x*block3d.y*block3d.z;
	//cudaMalloc(&blockResult, blockResultSize*sizeof(float));
	//blockResult_local = (float*)malloc(blockResultSize*sizeof(float));

	//d_scalarProduct<<<grid3d, block3d>>>(a, b, blockResult, w, h);
	//cudaDeviceSynchronize();
	float* a_local, *b_local;
	a_local = (float*)malloc(w*h * nc * sizeof(float));
	b_local = (float*)malloc(w*h * nc * sizeof(float));
	cudaMemcpy(a_local, a, w*h * nc * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(b_local, b, w*h * nc * sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_CHECK;

	//cudaMemcpy(blockResult_local, blockResult, blockResultSize*sizeof(float), cudaMemcpyDeviceToHost);
	//CUDA_CHECK;
	//cudaFree(blockResult);
	//CUDA_CHECK;

	long float result = 0.f;
	/**for (int i = 0; i < blockResultSize; i++) {
		result += blockResult_local[i];
	}**/
	for (int i = 0; i < w*h * nc; i++) {
		result += a_local[i] * b_local[i];
	}
	//free(blockResult_local);
	free(a_local);
	free(b_local);
	return result;
}

int fixedSizeFixedValuesTestcase() {
	cout << "-----------------------------" << endl;
	cout << "fixedSizeFixedValuesTestcase." << endl;
	const int w_small = 3;
	const int h_small = 3;
	const int w_big = 2 * w_small;
	const int h_big = 2 * h_small;
	const int nc = 3;
	const int n_small = w_small*h_small*nc;
	const int n_big = w_big*h_big*nc;
	float* u;
	float* Atv;
	float* v;
	float* Au;
	cudaMalloc(&u, n_big*sizeof(float));
	cudaMalloc(&Atv, n_big*sizeof(float));
	cudaMalloc(&v, n_small*sizeof(float));
	cudaMalloc(&Au, n_small*sizeof(float));

	dim3 block3d = dim3(16, 16, nc);
	dim3 grid3d = dim3((w_big + block3d.x - 1) / block3d.x, (h_big + block3d.y - 1) / block3d.y, 1);
	dim3 grid3d_small = dim3((w_small + block3d.x - 1) / block3d.x, (h_small + block3d.y - 1) / block3d.y, 1);

	initArrayFixValues<<<grid3d_small, block3d>>>(v, w_small, h_small);
	initArrayFixValues<<<grid3d, block3d>>>(u, w_big, h_big);

	execA_U(grid3d, grid3d_small, block3d, u, Au, w_small, w_big, h_small, h_big, nc);
	long float AuV = scalarProduct(block3d, grid3d_small, Au, v, w_small, h_small, nc);
	cout << "<Au, v> = " << AuV << endl;

	execAt_v(grid3d, block3d, v, Atv, w_small, w_big, h_small, h_big, nc);
	long float AtvU = scalarProduct(block3d, grid3d, Atv, u, w_big, h_big, nc);
	cout << "<(A^t)v, u> = " << AtvU << endl;

	cudaFree(u);
	cudaFree(Atv);
	cudaFree(v);
	cudaFree(Au);
	CUDA_CHECK;

	bool success = false;
	if (AuV == AtvU) {
		success = true;
		cout << "Test successfull." << endl;
	} else {
		cout << "Test failed!" << endl;
	}

	cout << "-----------------------------" << endl;
	if(success) return 0;
	return 1;
}

int main(int argc, char* argv[]) {
	cudaDeviceSynchronize();
	CUDA_CHECK;
	int totalTestCases = 1;
	int failedTestCases = 0;
	failedTestCases += fixedSizeFixedValuesTestcase();
	cout << endl << endl << "-----------------------------" << endl << "All test cases ran." << endl;
	cout << "Successfull: " << totalTestCases - failedTestCases << endl;
	cout << "Failed:      " << failedTestCases << endl;
	cout << "-----------------------------" << endl;
	system("pause");
	return 0;
}

