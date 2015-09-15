#include "flow_color.h"
#include <math.h>

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
	int y = threadIdx.y + blockDim.y * blockDim.y;

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
	int y = threadIdx.y + blockDim.y * blockDim.y;

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
	}
	else {
		// vector is to short for being color coded
		d_out[idx] = 0.5f * in_r;
		d_out[idx + w*h] = 0.5f * in_g;
		d_out[idx + 2 * w*h] = 0.5f * in_b;
	}
}