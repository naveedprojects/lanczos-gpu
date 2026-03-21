/**
 * cast_kernels.cu — FP64 ↔ FP32 conversion kernels for mixed-precision SpMV.
 */

#include <cuda_runtime.h>

__global__ void kernel_double_to_float(const double *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)in[i];
}

__global__ void kernel_float_to_double(const float *in, double *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (double)in[i];
}
