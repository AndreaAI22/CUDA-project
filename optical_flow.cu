#include <opencv2/opencv.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sobel.h"

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        std::exit(1);                                              \
    }                                                              \
} while(0)


void run_optical_flow(const unsigned char* buffer_current_frame, unsigned char* buffer_output_frame, int width, int heigth){
    
    static bool initialized = false;
    static unsigned char* buffer_previous_frame = nullptr;
    static cv::Ptr<cv::cuda::FarnebackOpticalFlow> object_algorithm;

    size_t nbytes = (size_t)width * (size_t)heigth;

    if(initialized == false){

        CUDA_CHECK(cudaMalloc((void**)&buffer_previous_frame, nbytes));
        CUDA_CHECK(cudaMemcpy(buffer_previous_frame, buffer_current_frame, nbytes, cudaMemcpyDeviceToDevice));

        object_algorithm = cv::cuda::FarnebackOpticalFlow::create();

        CUDA_CHECK(cudaMemset(buffer_output_frame, 0, nbytes)); //perche???

        initialized = true;
        return;

    }

    cv::cuda::GpuMat previous_frame(heigth, width, CV_8UC1, (void*)buffer_previous_frame, (size_t)width);
    cv::cuda::GpuMat current_frame(heigth, width, CV_8UC1, (void*)buffer_current_frame, (size_t)width);
    cv::cuda::GpuMat output_frame(heigth, width, CV_8UC1, (void*)buffer_output_frame, (size_t)width);
    cv::cuda::GpuMat d_flow; //se non specifico parametri è adattabile???

    object_algorithm->calc(previous_frame, current_frame, d_flow);

    std::vector<cv::cuda::GpuMat> xy;
    cv::cuda::split(d_flow, xy);

    cv::cuda::GpuMat d_mag, d_ang;
    cv::cuda::cartToPolar(xy[0], xy[1], d_mag, d_ang, false);

    cv::cuda::normalize(d_mag, output_frame, 0, 255, cv::NORM_MINMAX, CV_8U);

    CUDA_CHECK(cudaMemcpy(buffer_previous_frame, buffer_current_frame, nbytes, cudaMemcpyDeviceToDevice));


}