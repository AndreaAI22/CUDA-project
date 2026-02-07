#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "sobel.h"

__global__ void crop_kernel(const unsigned char* buffer_device_in, unsigned char* buffer_device_out, int roi_x, int roi_y, int roi_width, int roi_height, int image_width, int image_height){

    //compute the coordinates x, y of the pixel of output relativa to ROI(output)
    int x_o = threadIdx.x + blockDim.x * blockIdx.x;
    int y_o = threadIdx.y + blockDim.y * blockIdx.y;

    //menage of the bounds
    if(x_o >= roi_width || y_o >= roi_height){
        return;
    }

    int idx_o = y_o * roi_width + x_o;

    //compute the coordinates x,y relative of the pixel of input to read it 
    //traslation
    int x_i = x_o + roi_x;
    int y_i = y_o + roi_y;

    int idx_i = y_i * image_width + x_i;

    buffer_device_out[idx_o] = buffer_device_in[idx_i];

}


void run_crop_kernel(const unsigned char* buffer_device_in, unsigned char* buffer_device_out, int roi_x, int roi_y, int roi_width, int roi_height, int image_width, int image_height){

    dim3 block(16,16);
    dim3 grid((roi_width + block.x - 1) / block.x, (roi_height + block.y - 1) / block.y);
    crop_kernel<<<grid,block>>>(buffer_device_in, buffer_device_out, roi_x, roi_y, roi_width, roi_height, image_width, image_height);
    CUDA_CHECK(cudaGetLastError());
    
}