#pragma once

void run_edge_detection_sobel(const unsigned char* buffer_device_in, unsigned char* buffer_device_out, int width, int height, int limit);
void run_crop_mask_kernel(const unsigned char* buffer_device_in, unsigned char* buffer_device_out, int roi_x, int roi_y, int roi_width, int roi_height, int image_width, int image_height);
void run_crop_kernel(const unsigned char* buffer_device_in, unsigned char* buffer_device_out, int roi_x, int roi_y, int roi_width, int roi_height, int image_width, int image_height);
void run_scale_kernel(const unsigned char* buffer_device_in, unsigned char* buffer_device_out, int out_width, int out_height, int in_width, int in_height);