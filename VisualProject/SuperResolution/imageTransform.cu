#include "imageTransform.h"
#include "helper_math.h"
__constant__ float blurKernel[21];


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


__device__ void blur(int x, int y, int c, float* img, int w, int h){
	//TODO
}

__device__ void downsample(int x, int y, int c, float* in, int w, int h, float* out) {
	blur(x, y, c, in, w, h);

	int w_small = w / 2;
	int h_small = h / 2;
	int ind_in = x * 2 + y * 2 * w + w*h*c;
	int ind_out = x + y*w_small + c*w_small*h_small;
	out[ind_out] = in[ind_in];
}



__global__ void blur(float *in, float *out, int w, int h, float kernelDia){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;

	int idx = x + y * w + c * w * h;

	int radius = kernelDia / 2;

	int shIdxX = threadIdx.x + radius;
	int shIdxY = threadIdx.y + radius;
	int thIdx = threadIdx.x + threadIdx.y * blockDim.x;

	extern __shared__ float sh_data[];
	int sharedSizeX = blockDim.x + kernelDia - 1;
	int sharedSizeY = blockDim.y + kernelDia - 1;

	int x0 = blockDim.x * blockIdx.x;
	int y0 = blockDim.y * blockIdx.y;
	int x0s = x0 - radius;
	int y0s = y0 - radius;

	for (int sidx = thIdx; sidx < sharedSizeX*sharedSizeY; sidx += blockDim.x*blockDim.y) {
		int ix = clamp(x0s + sidx % sharedSizeX, 0, w - 1);
		int iy = clamp(y0s + sidx / sharedSizeX, 0, h - 1);
		sh_data[sidx] = in[ix + w * iy + c * w * h];
	}
	__syncthreads();

	// horizontal smooth
	if (x < w)	{
		float sum = 0.0f;
		for (int i = 0; i < kernelDia; i++){
			sum += blurKernel[i] * sh_data[shIdxX + i - radius + shIdxY * sharedSizeX];
		}
		sh_data[shIdxX + shIdxY * sharedSizeX] = sum;
	}
	__syncthreads();

	// vertical smooth
	if (x < w && y < h)	{
		float sum = 0.0f;
		for (int i = 0; i < kernelDia; i++){
			sum += blurKernel[i] * sh_data[shIdxX + (shIdxY + i - radius) * sharedSizeX];
		}
		out[idx] = sum;
	}
}


__global__ void upsample(float* in, float* out, int w, int h){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;
	int w_big = 2 * w;
	int h_big = 2 * h;
	if (x >= w_big || y >= h_big) return;

	int ind_in = x/2 + y/2*w + w*h*c;
	int ind_out = x + y * w_big + w_big * h_big * c;
	out[ind_out] = in[ind_in];

	//blurring has to be done at a later point, where global
	//synchronization between all threads (of all blocks) can be ensured
	//blur(x, y, c, out, w_big, h_big);
}

#define GK5_0 0.3434064786f
#define GK5_1 0.2426675967f
#define GK5_2 0.0856291639f
#define GK5_A3 1.4887526836f
#define GK5_A4 1.0936481794f

__global__ void gaussBlur5(float* in, float* out, int w, int h) {
	// shared memory for optimized memory access
	extern __shared__ float sdata[];

	// indices for image access
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;
	int wb = blockDim.x + 4;

	int sindex = threadIdx.x + 2 + (threadIdx.y + 2) * wb;
	int index = x + y * w + c * w * h;
	bool realPixel = (x >= 0 && x < w && y >= 0 && y < h);

	// fill shared memory (area covered by this block + 2 pixel of additional border)
	float accum;
	for (int si = threadIdx.x + threadIdx.y * blockDim.x; si < wb*(blockDim.y + 4); si += blockDim.x * blockDim.y) {
		int inX = blockIdx.x * blockDim.x - 2 + si % wb;
		int inY = blockIdx.y * blockDim.y - 2 + si / wb;
		accum = 0.0f;
		if (inX >= 0 && inX < w && inY >= 0 && inY < h) {
			accum = in[inX + inY * w + c * w * h];
		}
		sdata[si] = accum;
	}

	// wait until all threads have stored the image data
	__syncthreads();

	float accum2;
	if (realPixel) {
		// blur horizontally
		accum = sdata[sindex - 2] * GK5_2 + sdata[sindex - 1] * GK5_1 + sdata[sindex] * GK5_0 + sdata[sindex + 1] * GK5_1 + sdata[sindex + 2] * GK5_2;

		if (x == 0 || x == w - 1) {
			accum *= GK5_A3;
		} else if (x == 1 || x == w - 2) {
			accum *= GK5_A4;
		}
		// for the subsequent vertical blur two additional lines at top and bottom of the block have to be blurred as well
		if (y <= 1 || y >= h - 2) {
			int shiftIndex = sindex + (y > 1 ? 2 : -2) * wb;
			accum2 = sdata[shiftIndex - 2] * GK5_2 + sdata[shiftIndex - 1] * GK5_1 + sdata[shiftIndex] * GK5_0 + sdata[shiftIndex + 1] * GK5_1 + sdata[shiftIndex + 2] * GK5_2;
		}
	}

	// wait until all threads have computed the horizontal blur
	__syncthreads();

	if (realPixel) {
		// store blurred pixels into shared memory
		sdata[sindex] = accum;
		if (y <= 1 || y >= h - 2) {
			sdata[sindex + (y > 1 ? 2 : -2) * wb] = accum2;
		}
	}

	// wait until all threads have stored the horizontally blurred pixel values
	__syncthreads();

	if (realPixel) {
		// blur vertically
		accum = sdata[sindex - 2 * wb] * GK5_2 + sdata[sindex - wb] * GK5_1 + sdata[sindex] * GK5_0 + sdata[sindex + wb] * GK5_1 + sdata[sindex + 2 * wb] * GK5_2;

		if (y == 0 || y == h - 1) {
			accum *= GK5_A3;
		} else if (y == 1 || y == h - 2) {
			accum *= GK5_A4;
		}

		// store result in output image
		out[index] = accum;
	}
}