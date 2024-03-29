#include "projections.h"
#include <math.h>

__device__ float2 projL2(float2 x, float limit) {
	float l = x.x*x.x + x.y*x.y;
	if (l <= limit*limit) {
		return x;
	}

	l = limit/sqrtf(l);
	x.x *= l;
	x.y *= l;
	return x;
}

__device__ float projL1(float x, float limit) {
	if (x < -limit)
		return -limit;
	if (x > limit)
		return limit;
	return x;
}
