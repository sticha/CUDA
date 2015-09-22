#ifndef IMAGE_TRANSFORM_H_
#define IMAGE_TRANSFORM_H_

#include <cuda_runtime.h>

// x and y for the small image
__device__ void downsample(int x, int y);
// x and y for the big image
__device__ void upsample(int x, int y);

#endif
