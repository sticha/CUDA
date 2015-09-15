#include "flow_color.h"
#include <math.h>

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

	// compute angle
	float v2 = d_v2[idx];
	float angle = acosf(v2 / sqrtf(v2));
	if (d_v1[idx] < 0) {
		angle = 2 * PI - angle;
	}

	// get color index and color interpolant
	float colorInterp = angle * 3 / PI;
	int colorIdx = static_cast<int>(colorInterp);
	colorInterp -= colorIdx;

	// apply color scheme to output image
	const float intensities[] = {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
	d_out[idx] = intensities[colorIdx] + colorInterp * (intensities[(colorIdx + 1) % 6] - intensities[colorIdx]);
	d_out[idx + w*h] = intensities[(colorIdx + 2) % 6] + colorInterp * (intensities[(colorIdx + 3) % 6] - intensities[(colorIdx + 2) % 6]);
	d_out[idx + 2 * w*h] = intensities[(colorIdx + 4) % 6] + colorInterp * (intensities[(colorIdx + 5) % 6] - intensities[(colorIdx + 4) % 6]);
}