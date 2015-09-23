#include "imageTransform.h"

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

__global__ void upsample(float* in, float* out, int w, int h){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int c = threadIdx.z;
	int w_big = 2 * w;
	int h_big = 2 * h;
	if (x >= w_big || y >= h_big) return;

	int ind_in = x/2 + y/2*w + w*h*c;
	int ind_out = x + y * w_big + w_big * h_big * c;
	if (x % 2 || y % 2){
		out[ind_out] = 0.f;
	} else {
		out[ind_out] = in[ind_in];
	}

	//blurring has to be done at a later point, where global
	//synchronization between all threads (of all blocks) can be ensured
	//blur(x, y, c, out, w_big, h_big);
}
