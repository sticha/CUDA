// 1D Kernel
__constant__ float * blurKernel;
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
	int idx_big = x_big + y_big * w_big + c * w_big * h_big;


	float result;
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
			if (!offsX || i > 0) {
				sum += blurKernel[2 * j - offsX] * val;
			}
			// ignore most right value if a right pixel is evaluated
			if (offsX || i < kernelDia_Small - 1) {
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


__device__ float downsample(int x_small, int y_small, int c, float* in, int w_small, int h_small) {
	int x_big = x_small << 1;
	int y_big = y_small << 1;
	int w_big = w_small << 1;
	int h_big = h_small << 1;

	int idx_big = x_big + y_big * w_big + c * w_big * h_big;

	float result = 0.0f;

	for (int i = 0; i < kernelDia; i++) {
		float sum = 0.0f;
		for (int j = 0; j < kernelDia; j++) {
			int valIdx_X = clamp(x_big + j - kernelDia / 2, 0, w_big - 1);
			int valIdx_Y = clamp(y_big + i - kernelDia / 2, 0, h_big - 1);
			sum += blurKernel[j] * in[valIdx_X + valIdx_Y * w_big];
		}
		result += sum * blurKernel[i];
	}
	return result;
}