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

__device__ void upsample(int x, int y, int c, float* in, int w, int h, float* out){
	int w_big = 2 * w;
	int h_big = 2 * h;
	int ind_in = x/2 + y/2*w + w*h*c;
	int ind_out = x + y * w_big + w_big * h_big * c;
	if (x % 2 || y % 2){
		out[ind_out] = 0.f;
	} else {
		out[ind_out] = in[ind_in];
	}

	blur(x, y, c, out, w_big, h_big);
}
