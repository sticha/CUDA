#include "imageTransform.h"
#include "helper_math.h"
__device__ float blurKernel[5] = { 0.0219296448f, 0.2285121468f, 0.4991164165f, 0.2285121468f, 0.0219296448f };

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

__global__ void downsample(float* in_big, float* out_small, int w, int h, int w_small, int h_small) {
	// indices for low resolution image access
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;

	int x_big = x * 2;
	int y_big = y * 2;

	// return if pixel outside image
	if (x >= w_small || y >= h_small) {
		return;
	}

	// indices to access pixel
	int ind_big = x_big + y_big * w + c * w * h;
	int ind_small = x + y * w_small + c * w_small * h_small;

	out_small[ind_small] = 0.25f * in_big[ind_big] + 0.25f * in_big[ind_big+1] + 0.25f * in_big[ind_big+w] + 0.25f * in_big[ind_big+w+1];
}

__global__ void upsample(float* in_small, float* out_big, int w, int h, int w_small, int h_small) {
	// indices for high resolution image access
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;

	int x_small = x / 2;
	int y_small = y / 2;

	// return if pixel outside image
	if (x >= w || y >= h) {
		return;
	}

	// indices to access pixel
	int ind_big = x + y * w + c * w * h;
	int ind_small = x_small + y_small * w_small + c * w_small * h_small;

	//if ((x & 1) == 1 || (y & 1) == 1) {
	//	out_big[ind_big] = 0.0f;
	//} else {
		out_big[ind_big] = 0.25f * in_small[ind_small];
	//}
}

// Kernel to sample the input images up for initialization of the u_i
__global__ void initialUpsample(float* in, float* out, int w, int h, int w_small, int h_small) {
	// indices for image access
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;

	// return if pixel outside image
	if (x >= w || y >= h) {
		return;
	}

	// get pixel coordinates of input image
	int x_small = x * w_small / w;
	int y_small = y * h_small / h;

	// get pixel value of input image
	out[x + y * w + c * w * h] = in[x_small + y_small * w_small + c * w_small * h_small];
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
			float val = 0.25f * in[valIdx_X + valIdx_Y * w_small + w_small * h_small * c];
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

// sigma = 1.2
//#define GK5_0 0.3434064786f
//#define GK5_1 0.2426675967f
//#define GK5_2 0.0856291639f

// sigma = 0.8
//#define GK5_0 0.4991164165f
//#define GK5_1 0.2285121468f
//#define GK5_2 0.0219296448f

// sigma = 0.6
#define GK5_0 0.6638183293f
#define GK5_1 0.1655245666f
#define GK5_2 0.0025662686f

__global__ void gaussBlur5(float* in, float* out, int w, int h) {
	// shared memory for optimized memory access
	extern __shared__ float sdata[];

	// indices for image access
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;
	int wb = blockDim.x + 4;

	int sindex = threadIdx.x + 2 + (threadIdx.y + 2) * wb + wb * (blockDim.y + 4) * c;
	int index = x + y * w + c * w * h;
	bool realPixel = (x < w && y < h);

	// fill shared memory (area covered by this block + 2 pixel of additional border)
	float accum;
	for (int si = threadIdx.x + threadIdx.y * blockDim.x + c * blockDim.x * blockDim.y; si < wb*(blockDim.y + 4)*blockDim.z; si += blockDim.x * blockDim.y * blockDim.z) {
		int inX = min(w - 1, max(0, blockIdx.x * blockDim.x - 2 + (si % wb)));
		int inY = min(h - 1, max(0, blockIdx.y * blockDim.y - 2 + ((si / wb) % (blockDim.y + 4))));
		int inZ = si / (wb*(blockDim.y + 4));
		accum = in[inX + inY * w + inZ * w * h];
		sdata[si] = accum;
	}

	// wait until all threads have stored the image data
	__syncthreads();

	float accum2;
	if (realPixel) {
		// blur horizontally
		accum = sdata[sindex - 2] * GK5_2 + sdata[sindex - 1] * GK5_1 + sdata[sindex] * GK5_0 + sdata[sindex + 1] * GK5_1 + sdata[sindex + 2] * GK5_2;

		// for the subsequent vertical blur two additional lines at top and bottom of the block have to be blurred as well
		if (threadIdx.y <= 1 || threadIdx.y >= blockDim.y - 2) {
			int shiftIndex = sindex + (threadIdx.y > 1 ? 2 : -2) * wb;
			accum2 = sdata[shiftIndex - 2] * GK5_2 + sdata[shiftIndex - 1] * GK5_1 + sdata[shiftIndex] * GK5_0 + sdata[shiftIndex + 1] * GK5_1 + sdata[shiftIndex + 2] * GK5_2;
		}
	}

	// wait until all threads have computed the horizontal blur
	__syncthreads();

	if (realPixel) {
		// store blurred pixels into shared memory
		sdata[sindex] = accum;
		if (threadIdx.y <= 1 || threadIdx.y >= blockDim.y - 2) {
			sdata[sindex + (threadIdx.y > 1 ? 2 : -2) * wb] = accum2;
		}
	}

	// wait until all threads have stored the horizontally blurred pixel values
	__syncthreads();

	if (realPixel) {
		// blur vertically
		accum = sdata[sindex - 2 * wb] * GK5_2 + sdata[sindex - wb] * GK5_1 + sdata[sindex] * GK5_0 + sdata[sindex + wb] * GK5_1 + sdata[sindex + 2 * wb] * GK5_2;

		// store result in output image
		out[index] = accum;
	}
}

__global__ void blur(float *in, float *out, int w, int h, float kernelDia){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;

	int idx = x + y * w + c * w * h;

	int radius = kernelDia / 2;

	extern __shared__ float sh_data[];
	int sharedSizeX = blockDim.x + kernelDia - 1;
	int sharedSizeY = blockDim.y + kernelDia - 1;

	int shIdxX = threadIdx.x + radius;
	int shIdxY = threadIdx.y + radius;
	int shIdx = shIdxX + shIdxY * sharedSizeX;
	int thIdx = threadIdx.x + threadIdx.y * blockDim.x;

	

	int x0 = blockDim.x * blockIdx.x;
	int y0 = blockDim.y * blockIdx.y;
	int x0s = x0 - radius;
	int y0s = y0 - radius;

	bool isImgPixel = x < w && y < h;
	
	for (int sidx = thIdx; sidx < sharedSizeX*sharedSizeY; sidx += blockDim.x*blockDim.y) {
		int ix = clamp(x0s + sidx % sharedSizeX, 0, w - 1);
		int iy = clamp(y0s + sidx / sharedSizeX, 0, h - 1);
		sh_data[sidx] = in[ix + w * iy + c * w * h];
	}
	__syncthreads();

	// horizontal smooth
	float sum1 = 0.0f;
	float sum2 = 0.0f;
	if (isImgPixel)	{
		for (int i = 0; i < kernelDia; i++){
			sum1 += blurKernel[i] * sh_data[shIdx + i - radius];
		}
		if (threadIdx.x < radius || threadIdx.x >= blockDim.x - radius) {
			int shiftIndex = shIdx + (threadIdx.y >= radius ? radius : -radius) * sharedSizeX;
			for (int i = 0; i < kernelDia; i++) {
				sum2 += blurKernel[i] * sh_data[shiftIndex];
			}
		}
	}
	__syncthreads();

	if (isImgPixel) {
		// store blurred pixels into shared memory
		sh_data[shIdx] = sum1;
		if (threadIdx.y <= 1 || threadIdx.y >= blockDim.y - 2) {
			sh_data[shIdx + (threadIdx.y >= radius ? radius : -radius) * sharedSizeX] = sum2;
		}
	}

	__syncthreads();

	// vertical smooth
	if (isImgPixel)	{
		float sum = 0.0f;
		for (int i = 0; i < kernelDia; i++){
			sum += blurKernel[i] * sh_data[shIdxX + (shIdxY + i - radius) * sharedSizeX];
		}
		out[idx] = sum;
	}
}
