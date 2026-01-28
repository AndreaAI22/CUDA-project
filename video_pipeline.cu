#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

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

    unsigned char* buffer_host_in = frame_grayscale.data;

    //allocate buffer host for output and the two buffer in device(GPU)
    unsigned char* buffer_host_out = (unsigned char*) malloc(nbytes);
    if(buffer_host_out == nullptr){
        std::fprintf(stderr, "Error: malloc failed!\n");
        return 1;
    }

    unsigned char* buffer_device_in = nullptr;
    unsigned char* buffer_device_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&buffer_device_in, nbytes));
    CUDA_CHECK(cudaMalloc((void**)&buffer_device_out, nbytes));


    //now , we have acquired the properties of the frames of this video source
    //we can continue with the true elaboration

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

        //execution edge_detection_sobel kernel
        run_edge_detection_sobel(buffer_device_in, buffer_device_out, w, h, 50);
        

        //do the copy of the elaborated frame from DEVICE to HOST
        CUDA_CHECK(cudaMemcpy(buffer_host_out, buffer_device_out, nbytes, cudaMemcpyDeviceToHost));

        cv::Mat frame_output(h, w, CV_8UC1, buffer_host_out);

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