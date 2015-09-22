#include "projections.h"
#include <math.h>

__device__ float2 projL2(float2 x) {
	float l = x.x*x.x + x.y*x.y;
	if (l <= 1.f) {
		return x;
	}

	l = sqrtf(l);
	x.x /= l;
	x.y /= l;
	return x;
}

__device__ float projL1(float x, float gamma) {
	if (x < -gamma)
		return -gamma;
	if (x> gamma)
		return gamma;
	return x;
}
