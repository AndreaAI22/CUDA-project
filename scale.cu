#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include "sobel.h"

__global__ void scale_kernel(const unsigned char* buffer_device_in, unsigned char* buffer_device_out, int out_width, int out_height, int in_width, int in_height, float scale_factor_x, float scale_factor_y){


    int x_o = threadIdx.x + blockIdx.x * blockDim.x;
    int y_o = threadIdx.y + blockIdx.y * blockDim.y;

    if(x_o >= out_width || y_o >= out_height){
        return;
    }

    int idx_o = y_o * out_width + x_o;

    //i use the strategy nearest neighboor
    int x_i = (int)roundf(x_o * scale_factor_x);
    int y_i = (int)roundf(y_o * scale_factor_y);

    if (x_i >= in_width) {
        x_i = in_width - 1;
    }
    if (y_i >= in_height) {
        y_i = in_height - 1;
    }

    int idx_i = y_i * in_width + x_i;

    buffer_device_out[idx_o] = buffer_device_in[idx_i];

}

void run_scale_kernel(const unsigned char* buffer_device_in, unsigned char* buffer_device_out, int out_width, int out_height, int in_width, int in_height, float scale_factor_x, float scale_factor_y){
    dim3 block(16,16);
    dim3 grid((out_width + block.x - 1) / block.x, (out_height + block.y - 1) / block.y);
    scale_kernel<<<grid, block>>>(buffer_device_in, buffer_device_out, out_width, out_height, in_width, in_height, scale_factor_x, scale_factor_y);
    CUDA_CHECK(cudaGetLastError());
}