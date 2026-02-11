#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        std::exit(1);                                              \
    }                                                              \
} while(0)

void run_edge_detection_sobel(const unsigned char* buffer_device_in, unsigned char* buffer_device_out, int width, int height, int limit);
void run_crop_mask_kernel(const unsigned char* buffer_device_in, unsigned char* buffer_device_out, int roi_x, int roi_y, int roi_width, int roi_height, int image_width, int image_height);
void run_crop_kernel(const unsigned char* buffer_device_in, unsigned char* buffer_device_out, int roi_x, int roi_y, int roi_width, int roi_height, int image_width, int image_height);
void run_scale_kernel(const unsigned char* buffer_device_in, unsigned char* buffer_device_out, int out_width, int out_height, int in_width, int in_height, float scale_factor_x, float scale_factor_y);
void run_rotate_kernel(const unsigned char* buffer_device_in, unsigned char* buffer_device_out, int image_width, int image_height, float c, float s);
void run_optical_flow(const unsigned char* buffer_current_frame, unsigned char* buffer_output_frame, int width, int heigth);
    
