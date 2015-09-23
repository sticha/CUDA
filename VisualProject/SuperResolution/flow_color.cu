#include "flow_color.h"
#include <math.h>

#define PI 3.1415926535f
#define EPSILON 0.0001

__device__ float d_getAngleFromVector(float v1, float v2) {
	// compute angle in radians between motion vector v and (0, 1)
	// the component v2 is assumed to be normalized w.r.t. the original vector v
	float angle = acosf(v2);
	if (v1 < 0) {
		angle = 2 * PI - angle;
	}
	return angle;
}

__global__ void createColorCoding(float* d_v1, float* d_v2, float* d_out, int w, int h) {
	// get current thread index (x, y)
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	// return if coordinate (x, y) not inside image
	if (x >= w || y >= h) {
		return;
	}

	// index for access image pixel
	int idx = x + w * y;

	// compute vector length
	float v1 = d_v1[idx];
	float v2 = d_v2[idx];
	float v_len = sqrtf(v1*v1 + v2*v2);

	if (v_len > EPSILON) {
		// compute angle
		float angle = d_getAngleFromVector(v1, v2 / v_len);

		// use weighted v_len for speed
		v_len *= 0.3f;

		// get color index and color interpolant
		float colorInterp = angle * 3 / PI;
		int colorIdx = static_cast<int>(colorInterp);
		colorInterp -= colorIdx;

		// apply color scheme to output image
		const float intensities[] = { 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };
		float red = intensities[colorIdx] + colorInterp * (intensities[(colorIdx + 1) % 6] - intensities[colorIdx]);
		float green = intensities[(colorIdx + 2) % 6] + colorInterp * (intensities[(colorIdx + 3) % 6] - intensities[(colorIdx + 2) % 6]);
		float blue = intensities[(colorIdx + 4) % 6] + colorInterp * (intensities[(colorIdx + 5) % 6] - intensities[(colorIdx + 4) % 6]);
		d_out[idx] = fminf(1.0f, v_len*red);
		d_out[idx + w*h] = fminf(1.0f, v_len*green);
		d_out[idx + 2 * w*h] = fminf(1.0f, v_len*blue);
	} else {
		// vector to short for beeing color coded
		d_out[idx] = 0.0f;
		d_out[idx + w*h] = 0.0f;
		d_out[idx + 2 * w*h] = 0.0f;
	}
}

__global__ void createColorCoding(float* d_in, float* d_v1, float* d_v2, float* d_out, int w, int h, int nc) {
	// get current thread index (x, y)
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	// return if coordinate (x, y) not inside image
	if (x >= w || y >= h) {
		return;
	}

	// index for access image pixel
	int idx = x + w * y;

	// compute vector length
	float v1 = d_v1[idx];
	float v2 = d_v2[idx];
	float v_len = sqrtf(v1*v1 + v2*v2);

	// get input image color values
	float in_r = d_in[idx];
	float in_g = in_r;
	float in_b = in_r;
	if (nc == 3) {
		in_g = d_in[idx + w*h];
		in_b = d_in[idx + 2 * w*h];
	}

	if (v_len > EPSILON) {
		// compute angle
		float angle = d_getAngleFromVector(v1, v2 / v_len);

		// use weighted v_len for speed
		v_len *= 0.3f;

		// get color index and color interpolant
		float colorInterp = angle * 3 / PI;
		int colorIdx = static_cast<int>(colorInterp);
		colorInterp -= colorIdx;

		// apply color scheme to output image (merge with input image data)
		const float intensities[] = { 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };
		float red = intensities[colorIdx] + colorInterp * (intensities[(colorIdx + 1) % 6] - intensities[colorIdx]);
		float green = intensities[(colorIdx + 2) % 6] + colorInterp * (intensities[(colorIdx + 3) % 6] - intensities[(colorIdx + 2) % 6]);
		float blue = intensities[(colorIdx + 4) % 6] + colorInterp * (intensities[(colorIdx + 5) % 6] - intensities[(colorIdx + 4) % 6]);
		d_out[idx] = fminf(1.0f, 0.5f*v_len*red + 0.5f*in_r);
		d_out[idx + w*h] = fminf(1.0f, 0.5f*v_len*green + 0.5f*in_g);
		d_out[idx + 2 * w*h] = fminf(1.0f, 0.5f*v_len*blue + 0.5f*in_b);
	} else {
		// vector is to short for being color coded
		d_out[idx] = 0.5f * in_r;
		d_out[idx + w*h] = 0.5f * in_g;
		d_out[idx + 2 * w*h] = 0.5f * in_b;
	}
}

__global__ void createColorCoding(float* d_v1, float* d_v2, float* d_out, int w, int h, int border) {
	// get current thread index (x, y)
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	// return if coordinate (x, y) not inside image
	if (x >= w || y >= h) {
		return;
	}

	// index for access image pixel of image with border
	int idxb = x + w * y;
	// index for access image pixel inside image without border
	int idx = (x-border) + (w-2*border) * (y-border);

	// scale factor
	float scale = 0.4f;

	// compute vector length
	float v1, v2;
	if (x < border || x >= w - border || y < border || y >= h - border) {
		v1 = (x - w / 2.0f) / (fminf(w, h) * scale / 3.0f);
		v2 = (y - h / 2.0f) / (fminf(w, h) * scale / 3.0f);
	} else {
		v1 = d_v1[idx];
		v2 = d_v2[idx];
	}
	float v_len = sqrtf(v1*v1 + v2*v2);

	if (v_len > EPSILON) {
		// compute angle
		float angle = d_getAngleFromVector(v1, v2 / v_len);

		// use weighted v_len for speed
		v_len *= scale;

		// get color index and color interpolant
		float colorInterp = angle * 3 / PI;
		int colorIdx = static_cast<int>(colorInterp);
		colorInterp -= colorIdx;

		// apply color scheme to output image
		const float intensities[] = { 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };
		float red = intensities[colorIdx] + colorInterp * (intensities[(colorIdx + 1) % 6] - intensities[colorIdx]);
		float green = intensities[(colorIdx + 2) % 6] + colorInterp * (intensities[(colorIdx + 3) % 6] - intensities[(colorIdx + 2) % 6]);
		float blue = intensities[(colorIdx + 4) % 6] + colorInterp * (intensities[(colorIdx + 5) % 6] - intensities[(colorIdx + 4) % 6]);
		d_out[idxb] = fminf(1.0f, v_len*red);
		d_out[idxb + w*h] = fminf(1.0f, v_len*green);
		d_out[idxb + 2 * w*h] = fminf(1.0f, v_len*blue);
	} else {
		// vector to short for beeing color coded
		d_out[idxb] = 0.0f;
		d_out[idxb + w*h] = 0.0f;
		d_out[idxb + 2 * w*h] = 0.0f;
	}
}

__global__ void createColorCoding(float* d_in, float* d_v1, float* d_v2, float* d_out, int w, int h, int nc, int border) {
	// get current thread index (x, y)
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	// return if coordinate (x, y) not inside image
	if (x >= w || y >= h) {
		return;
	}

	bool isBorder = (x < border || x >= w - border || y < border || y >= h - border);

	// width without border
	int wfree = w - 2 * border;
	// height without border
	int hfree = h - 2 * border;

	// index for access image pixel of image with border
	int idxb = x + w * y;
	// index for access image pixel inside image without border
	int idx = (x - border) + wfree * (y - border);

	// scale factor
	float scale = 0.4f;

	// compute vector length
	float v1, v2;
	if (isBorder) {
		v1 = (x - w / 2.0f) / (fminf(w, h) * scale / 3.0f);
		v2 = (y - h / 2.0f) / (fminf(w, h) * scale / 3.0f);
	} else {
		v1 = d_v1[idx];
		v2 = d_v2[idx];
	}
	float v_len = sqrtf(v1*v1 + v2*v2);

	// get input image color values
	float in_r, in_g, in_b;
	if (!isBorder) {
		in_r = d_in[idx];
		in_g = in_r;
		in_b = in_r;
		if (nc == 3) {
			in_g = d_in[idx + wfree*hfree];
			in_b = d_in[idx + 2 * wfree*hfree];
		}
	}
	if (v_len > EPSILON) {
		// compute angle
		float angle = d_getAngleFromVector(v1, v2 / v_len);

		// use weighted v_len for speed
		v_len *= scale;

		// get color index and color interpolant
		float colorInterp = angle * 3 / PI;
		int colorIdx = static_cast<int>(colorInterp);
		colorInterp -= colorIdx;

		// apply color scheme to output image (merge with input image data)
		const float intensities[] = { 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };
		float red = intensities[colorIdx] + colorInterp * (intensities[(colorIdx + 1) % 6] - intensities[colorIdx]);
		float green = intensities[(colorIdx + 2) % 6] + colorInterp * (intensities[(colorIdx + 3) % 6] - intensities[(colorIdx + 2) % 6]);
		float blue = intensities[(colorIdx + 4) % 6] + colorInterp * (intensities[(colorIdx + 5) % 6] - intensities[(colorIdx + 4) % 6]);
		if (isBorder) {
			d_out[idxb] = fminf(1.0f, v_len*red);
			d_out[idxb + w*h] = fminf(1.0f, v_len*green);
			d_out[idxb + 2 * w*h] = fminf(1.0f, v_len*blue);
		} else {
			d_out[idxb] = fminf(1.0f, 0.5f*v_len*red + 0.5f*in_r);
			d_out[idxb + w*h] = fminf(1.0f, 0.5f*v_len*green + 0.5f*in_g);
			d_out[idxb + 2 * w*h] = fminf(1.0f, 0.5f*v_len*blue + 0.5f*in_b);
		}
	} else {
		// vector is to short for being color coded
		d_out[idxb] = 0.5f * in_r;
		d_out[idxb + w*h] = 0.5f * in_g;
		d_out[idxb + 2 * w*h] = 0.5f * in_b;
	}
}