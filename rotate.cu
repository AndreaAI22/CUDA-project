#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>

#include "sobel.h"

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        std::exit(1);                                              \
    }                                                              \
} while(0)

__global__ void rotate_kernel(const unsigned char* buffer_device_in, unsigned char* buffer_device_out, int image_width, int image_height, float rotate_angle){

    //compute the coordinate of pixel output
    int x_o = threadIdx.x + blockDim.x * blockIdx.x;
    int y_o = threadIdx.y + blockDim.y * blockIdx.y;

    if(x_o >= image_width || y_o >= image_height){
        //index out of bound; thread not necessary
        return;
    }

    int idx_o = y_o * image_width + x_o;

    //convert grades to radius
    float theta = (rotate_angle * 3.14f) / 180.0f;

    //compute center of the image
    float center_x = (float)(image_width - 1) / 2.0f;
    float center_y = (float)(image_height - 1) / 2.0f;

    float x_o_center = (float)x_o - center_x;
    float y_o_center = (float)y_o - center_y;

    //i use the - to apply the original formula,and not the inversa
    float x_i_center = x_o_center * cosf(-theta) - y_o_center * sinf(-theta);
    float y_i_center = x_o_center * sinf(-theta) + y_o_center * cosf(-theta);

    //now i have the coordinates of the pixel of input that i have to read
    int x_i = (int)roundf(x_i_center + center_x);
    int y_i = (int)roundf(y_i_center + center_y);


    if(x_i >= image_width || y_i >= image_height || x_i < 0 || y_i < 0 ){
        buffer_device_out[idx_o] = 0;
    }else{
        int idx_i = y_i * image_width + x_i;
        buffer_device_out[idx_o] = buffer_device_in[idx_i];
    }

}



void run_rotate_kernel(const unsigned char* buffer_device_in, unsigned char* buffer_device_out, int image_width, int image_height, float rotate_angle){
    dim3 block(16,16);
    dim3 grid((image_width + block.x - 1) / block.x, (image_height + block.y - 1) / block.y);
    rotate_kernel<<<grid, block>>>(buffer_device_in, buffer_device_out, image_width, image_height, rotate_angle);
    CUDA_CHECK(cudaGetLastError());
}