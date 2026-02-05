#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

#include "sobel.h"

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        std::exit(1);                                              \
    }                                                              \
} while(0)


int computeNumberBlocks(int n_elems, int n_threads){
    if(n_elems % n_threads == 0){
        return n_elems / n_threads;
    }else{
        return (n_elems / n_threads) +1;
    }
}



int main(){

    int roi_x = 300;
    int roi_y = 300;
    int roi_width = 800;
    int roi_height = 400;

    int current_width;
    int current_height;

    size_t nbytes_out = 0;


    //get the input to decide which operations must be execute
    int code_operation = 0;
    std::printf("Select the operation: \n");
    std::printf("Edge Detection (Sobel) = 1\nCrop = 2\nScale = 3\nRotate = 4\nOptical Flow = 5\n -->  ");
    std::cin >> code_operation;

    //connection to the video source (in this case is webcam)
    cv::VideoCapture cap(0);
    if (cap.isOpened() == false){
        std::fprintf(stderr, "Error: the video source can't be open!\n");
        return 1;
    }

    
    cv::Mat frame_original, frame_grayscale;

    //read a single frame to know the resolution and other properties

    cap >> frame_original;
    if(frame_original.empty()){
        std::fprintf(stderr, "Error: the frame can't be readen!\n");
        return 1;
    }

    //turn the frame in grayscale to make easier the computation
    cv::cvtColor(frame_original, frame_grayscale, cv::COLOR_BGR2GRAY);

    //check if the buffer is continuous (if there is padding)
    if(frame_grayscale.isContinuous() == false){
        frame_grayscale = frame_grayscale.clone();
    }

    int w = frame_grayscale.cols;
    int h = frame_grayscale.rows;
    int channels = frame_grayscale.channels();
    size_t nbytes = (size_t)w * (size_t)h * (size_t)channels;
    
    //in this context where the format pixel is grayscale , 1 pixel(element) = 1 byte
    int n_elements = (int)nbytes;


    if(code_operation == 1){
        //the user chooses Edge Detection
        current_width = w;
        current_height = h;
        nbytes_out = nbytes;

    }
    else if(code_operation == 2){
        //the user chooses Crop
        current_width = roi_width;
        current_height = roi_height;
        nbytes_out = current_width * current_height;
    }
    else if(code_operation == 3){
        //the user chooses Scale
        current_width = w/3;
        current_height = h/3;
        nbytes_out = current_width * current_height ;
    }
    else if(code_operation == 4){
        //the user chooses Rotate
        current_width = w;
        current_height = h;
        nbytes_out = nbytes;
    }
    else if(code_operation == 5){
        //the user chooses Optical Flow
        current_width = w;
        current_height = h;
        nbytes_out = nbytes;
    }
    else{
        //invalid operation
        std::fprintf(stderr, "ERROR: Invalid OPeration\n");
        return 1;
    }

    unsigned char* buffer_host_in = frame_grayscale.data;

    //allocate buffer host for output and the two buffer in device(GPU)
    unsigned char* buffer_host_out = (unsigned char*) malloc(nbytes_out);
    if(buffer_host_out == nullptr){
        std::fprintf(stderr, "Error: malloc failed!\n");
        return 1;
    }

    unsigned char* buffer_device_in = nullptr;
    unsigned char* buffer_device_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&buffer_device_in, nbytes));
    CUDA_CHECK(cudaMalloc((void**)&buffer_device_out, nbytes_out));

    while(true){

        cap >> frame_original;
        if(frame_original.empty()){
            break;
        }

        cv::cvtColor(frame_original, frame_grayscale, cv::COLOR_BGR2GRAY);

        if(frame_grayscale.isContinuous() == false){
            frame_grayscale = frame_grayscale.clone();
        }

        //to menage the change of resolution of the camera
        if (frame_grayscale.cols != w || frame_grayscale.rows != h){
            break;
        }

        buffer_host_in = frame_grayscale.data;

        //do the copy of the frame from HOST to DEVICE
        CUDA_CHECK(cudaMemcpy(buffer_device_in, buffer_host_in, nbytes, cudaMemcpyHostToDevice));

        if (code_operation == 1) {
            run_edge_detection_sobel(buffer_device_in, buffer_device_out, w, h, 50);
        } else if (code_operation == 2) {
            run_crop_kernel(buffer_device_in, buffer_device_out, roi_x, roi_y, roi_width, roi_height, w, h);
        }else if (code_operation == 3) {
            run_scale_kernel(buffer_device_in, buffer_device_out, current_width, current_height, w, h);
        }else if(code_operation == 4){
            run_rotate_kernel(buffer_device_in, buffer_device_out, current_width, current_height, 180);
        }
        else if(code_operation == 5){
            run_optical_flow(buffer_device_in, buffer_device_out, current_width, current_height);
        }
        else{
            fprintf(stderr, "ERROR: Invalid operation!\n");
            break;
        }

        //do the copy of the elaborated frame from DEVICE to HOST
        CUDA_CHECK(cudaMemcpy(buffer_host_out, buffer_device_out, nbytes_out, cudaMemcpyDeviceToHost));

        cv::Mat frame_output(current_height, current_width, CV_8UC1, buffer_host_out);

        //display to screen
        cv::imshow("Result ", frame_output);

        int code = cv::waitKey(1);
        if(code == 27){
            break;
        }

    }

    CUDA_CHECK(cudaFree(buffer_device_in));
    CUDA_CHECK(cudaFree(buffer_device_out));
    free(buffer_host_out);

    return 0;
}