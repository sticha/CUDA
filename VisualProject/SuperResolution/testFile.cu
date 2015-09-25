#include"helper.h"

__constant__ float blurKernel[21];
const int kernelDia = 5;
const int kernelDia_Small = kernelDia / 2 + 1;

__device__ int clamp(int val, int val_min, int val_max) {
	if (val < val_min)
		return val_min;
	if (val > val_max)
		return val_max;
	return val;
}

__device__ float upsample(float* in, int x_big, int y_big, int c, int w_big, int h_big) {
	// coordinates and size in the small image
	int x_small = x_big >> 1;
	int y_small = y_big >> 1;
	int w_small = w_big >> 1;
	int h_small = h_big >> 1;

	// calculate indices
	int idx_small = x_small + y_small * w_small + c * w_small * h_small;


	float result = 0.0f;
	// Offset value, zero for left pixel, one for right pixels
	int offsX = x_big % 2;
	// Offset value, zero for top pixels, one for bottom pixels
	int offsY = y_big % 2;
	for (int i = 0; i < kernelDia_Small; i++) {
		float sum = 0.0f;
		for (int j = 0; j < kernelDia_Small; j++) {
			// only get values inside the image
			int valIdx_X = clamp(x_small + j - kernelDia_Small / 2, 0, w_small - 1);
			int valIdx_Y = clamp(y_small + i - kernelDia_Small / 2, 0, h_small - 1);
			float val = in[valIdx_X + valIdx_Y * w_small];
			// ignore most left value if a left pixel is evaluated
			if (!offsX || j > 0) {
				sum += blurKernel[2 * j - offsX] * val;
			}
			// ignore most right value if a right pixel is evaluated
			if (offsX || j < kernelDia_Small - 1) {
				sum += blurKernel[2 * j + 1 - offsX] * val;
			}
		}
		// horizontal smooth
		// ignore most top value if a bottom pixel is evaluated
		if (!offsY || i > 0) {
			result += blurKernel[2 * i - offsY] * sum;
		}
		// ignore most bottom value if a top pixel is evaluated
		if (offsY || i < kernelDia_Small - 1) {
			result += blurKernel[2 * i + 1 - offsY] * sum;
		}
	}

	return result;
}

__global__ void testKernel(float * d_in, float * d_out1, float * d_out2, int w, int h){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= w || y >= h)
		return;

	int idx = x + y * w;

	d_out1[idx] = upsample(d_in, x, y, 0, w, h);
	d_out2[idx] = d_in[x / 2 + y / 2 * w / 2];

}

float getKernel(float * kernel, float sigma, int diameter){
	float sum = 0.0f;
	for (int y = 0; y < diameter; y++){
		int b = y - diameter / 2;
		float val = expf(-(b*b) / (2 * sigma*sigma));
		kernel[y] = val;
		sum += val;
	}
	return sum;
}

void getNormalizedKernel(float * kernel, float sigma, int diameter){
	float sum = getKernel(kernel, sigma, diameter);

	for (int i = 0; i < diameter; i++){
		kernel[i] /= sum;
	}
}

void testFunction(float * in, float * out1, float * out2, int w, int h){
	float kernel[kernelDia];
	getNormalizedKernel(kernel, 1.0f, kernelDia);
	float * d_in, *d_out1, *d_out2;
	cudaMalloc(&d_in, w / 2 * h / 2 * sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_out1, w *h*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_out2, w *h*sizeof(float));
	CUDA_CHECK;
	cudaMemcpy(d_in, in, w / 2 * h / 2 * sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMemcpyToSymbol(blurKernel, kernel, kernelDia*sizeof(float));
	CUDA_CHECK;

	dim3 block2d = dim3(16, 16, 1);
	dim3 grid2d = dim3((w + block2d.x - 1) / block2d.x, (h + block2d.y - 1) / block2d.y, 1);

	testKernel << <grid2d, block2d >> >(d_in, d_out1, d_out2, w, h);
	cudaDeviceSynchronize();
	CUDA_CHECK;

	cudaMemcpy(out1, d_out1, w*h*sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_CHECK;
	cudaMemcpy(out2, d_out2, w*h*sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_CHECK;

	cudaFree(d_in);
	cudaFree(d_out1);
	cudaFree(d_out2);
}