#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>

#include "sobel.h"

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        std::exit(1);                                              \
    }                                                              \
} while(0)

__global__ void scale_kernel(const unsigned char* buffer_device_in, unsigned char* buffer_device_out, int out_width, int out_height, int in_width, int in_height){

    //i use the strategy nearest neighboor

    int x_o = threadIdx.x + blockIdx.x * blockDim.x;
    int y_o = threadIdx.y + blockIdx.y * blockDim.y;

    if(x_o >= out_width || y_o >= out_height){
        return;
    }

    int idx_o = y_o * out_width + x_o;

    float scale_factor_x = (float)in_width / (float)out_width;
    float scale_factor_y = (float)in_height / (float)out_height;

    int x_i = (int)(x_o * scale_factor_x);
    int y_i = (int)(y_o * scale_factor_y);

    if (x_i >= in_width) {
        x_i = in_width - 1;
    }
    if (y_i >= in_height) {
        y_i = in_height - 1;
    }

    int idx_i = y_i * in_width + x_i;

    buffer_device_out[idx_o] = buffer_device_in[idx_i];

}

void run_scale_kernel(const unsigned char* buffer_device_in, unsigned char* buffer_device_out, int out_width, int out_height, int in_width, int in_height){
    dim3 block(16,16);
    dim3 grid((out_width + block.x - 1) / block.x, (out_height + block.y - 1) / block.y);
    scale_kernel<<<grid, block>>>(buffer_device_in, buffer_device_out, out_width, out_height, in_width, in_height);
    CUDA_CHECK(cudaGetLastError());
}