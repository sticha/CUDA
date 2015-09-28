#include "imageTransform.h"
#include "helper_math.h"
__constant__ float blurKernel[5];
const int kernelDia = 5;
const int kernelDia_Small = kernelDia / 2 + 1;

float getKernel(float* kernel, float sigma, int diameter){
	float sum = 0.0f;
	for (int y = 0; y < diameter; y++){
		int b = y - diameter / 2;
		float val = expf(-(b*b) / (2 * sigma*sigma));
		kernel[y] = val;
		sum += val;
	}
	return sum;
}

void getNormalizedKernel(float* kernel, float sigma, int diameter){
	float sum = getKernel(kernel, sigma, diameter);

	for (int i = 0; i < diameter; i++){
		kernel[i] /= sum;
	}
}

__global__ void downsample(float* in_big, float* out_small, int w, int h) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;
	int w_small = w / 2;
	int h_small = h / 2;
	if (x >= w_small || y >= h_small) return;

	int ind_big = x * 2 + y * 2 * w + w*h*c;
	int ind_small = x + y*w_small + c*w_small*h_small;
	out_small[ind_small] = in_big[ind_big];
}





__global__ void upsample(float* in_small, float* out_big, int w, int h){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;
	int w_big = 2 * w;
	int h_big = 2 * h;
	if (x >= w_big || y >= h_big) return;

	int ind_big = x + y * w_big + w_big * h_big * c;
	if (x % 2 == 1 || y % 2 == 1) {
		out_big[ind_big] = 0.f;
	} else {
		int ind_small = x/2 + (y/2)*w + w*h*c;
		out_big[ind_big] = in_small[ind_small];
	}
}

__device__ float d_downsample(float* in, int x_small, int y_small, int c, int w_small, int h_small) {
	int x_big = x_small << 1;
	int y_big = y_small << 1;
	int w_big = w_small << 1;
	int h_big = h_small << 1;

	float result = 0.0f;

	for (int i = 0; i < kernelDia; i++) {
		float sum = 0.0f;
		for (int j = 0; j < kernelDia; j++) {
			int valIdx_X = clamp(x_big + j - kernelDia / 2, 0, w_big - 1);
			int valIdx_Y = clamp(y_big + i - kernelDia / 2, 0, h_big - 1);
			sum += blurKernel[j] * in[valIdx_X + valIdx_Y * w_big + w_big * h_big * c];
		}
		result += sum * blurKernel[i];
	}
	return result;
}

__device__ float d_upsample(float* in, int x_big, int y_big, int c, int w_big, int h_big) {
	// coordinates and size in the small image
	int x_small = x_big >> 1;
	int y_small = y_big >> 1;
	int w_small = w_big >> 1;
	int h_small = h_big >> 1;

	float result = 0.0f;
	// Offset value, zero for left pixel, one for right pixels
	int offsX = x_big % 2;
	// Offset value, zero for top pixels, one for bottom pixels
	int offsY = y_big % 2;
	for (int i = 0; i < kernelDia_Small; i++) {
		float sum = 0.0f;
		int valIdx_Y = clamp(y_small + i - kernelDia_Small / 2, 0, h_small - 1);
		for (int j = 0; j < kernelDia_Small; j++) {
			// only get values inside the image
			int valIdx_X = clamp(x_small + j - kernelDia_Small / 2, 0, w_small - 1);
			float val = in[valIdx_X + valIdx_Y * w_small + w_small * h_small * c];
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

// For a 1 dimensional 5 pixel gaussian blur kernel with sigma = 1.2 you have the weights [GK5_2, GK5_1, GK5_0, GK5_1, GK5_2]
// Close to a border one part of the kernel might be unused so that the weights have to be renormalized (sum(w_i) = 1)
// Possible partial kernels with 4 or 3 pixels:
// [GK5_1, GK5_0, GK5_1, GK5_2] * GK5_AREA_4, [GK5_2, GK5_1, GK5_0, GK5_1] * GK5_AREA_4,
// [GK5_0, GK5_1, GK5_2] * GK5_AREA_3, [GK5_2, GK5_1, GK5_0] * GK5_AREA_3

#define GK5_0 0.3434064786f
#define GK5_1 0.2426675967f
#define GK5_2 0.0856291639f
#define GK5_AREA_3 1.4887526834f
#define GK5_AREA_4 1.0936481792f

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
			accum *= GK5_AREA_3;
		} else if (x == 1 || x == w - 2) {
			accum *= GK5_AREA_4;
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
			accum *= GK5_AREA_3;
		} else if (y == 1 || y == h - 2) {
			accum *= GK5_AREA_4;
		}

		// store result in output image
		out[index] = accum;
	}
}

/*__global__ void blur(float *in, float *out, int w, int h, float kernelDia){
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
}*/
