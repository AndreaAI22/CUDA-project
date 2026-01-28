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


__global__ void edge_detection_kernel(const unsigned char* buffer_device_in, unsigned char* buffer_device_out, int width, int height, int limit){

    //kernel launch in 2d (the grid of thread is in 2d)
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    //cause of there are often more thread than the necessary , the extra threads
    // could lead to having index out of bounds so must do this control
    if (x >= width || y >= height) return;

    //compute the index to access to the pixel/element that this thread have to compute
    int idx = y * width + x;

    //if the pixel is a border pixel then don't compute it 
    if (x == 0 || x == width-1 || y == 0 || y == height-1){
        buffer_device_out[idx] = buffer_device_in[idx];
        return;
    }

    //use int instead of unsigned char because, when i will use
    // these variables to compute Gx with the weighted sum there will be
    // the risk of overflow , cause the result can be greater than 255
    // and also , to get and store the neighboor pixels i could use 
    // an array of 9 elements and set it in a for loop with the expression
    // array[i] = buffer_device_in[(y-1+(i / 3))*width + (x - 1 + (i % 3))]
    // but cause of / and % , this operations could be expensive so
    // i read and store the pixels manually
    int p00 = buffer_device_in[(y-1)*width + (x-1)];
    int p01 = buffer_device_in[(y-1)*width + x];
    int p02 = buffer_device_in[(y-1)*width + (x+1)];
    int p10 = buffer_device_in[y*width + (x-1)];
    int p11 = buffer_device_in[y*width + x];
    int p12 = buffer_device_in[y*width + (x+1)];
    int p20 = buffer_device_in[(y+1)*width + (x-1)];
    int p21 = buffer_device_in[(y+1)*width + x];
    int p22 = buffer_device_in[(y+1)*width + (x+1)];

    //compute Gx and Gy
    int Gx = (-1 * p00) + (1 * p02) + (-2 * p10) + (2 * p12) + (-1 * p20) + (1 * p22);
    int Gy = (-1 * p00) + (-2 * p01) + (-1 * p02) + (1 * p20) + (2 * p21) + (1 * p22);

    int mag = abs(Gx) + abs(Gy);

    if (mag > 255){
        mag = 255;
    }

    if (mag >= limit){
        //then the pixel is an edge, and i display it in white
        buffer_device_out[idx] = 255;
    }else{
        //else the pixel isn't an edge
        //buffer_device_out[idx] = buffer_device_in[idx];
        buffer_device_out[idx] = 0;
    }

}


void run_edge_detection_sobel(const unsigned char* buffer_device_in, unsigned char* buffer_device_out, int width, int height, int limit){
    dim3 block(16,16);
    dim3 grid((width+15)/16, (height+15)/16);
    edge_detection_kernel<<<grid,block>>>(buffer_device_in, buffer_device_out, width, height, limit);
    CUDA_CHECK(cudaGetLastError());
}