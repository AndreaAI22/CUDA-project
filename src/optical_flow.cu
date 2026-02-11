#include <opencv2/opencv.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>
#include <cuda_runtime.h>
#include "cuda_common.h"

void run_optical_flow(const unsigned char* buffer_current_frame, unsigned char* buffer_output_frame, int width, int heigth){
    
    static bool initialized = false;
    static unsigned char* buffer_previous_frame = nullptr;
    static cv::Ptr<cv::cuda::FarnebackOpticalFlow> object_algorithm;

    size_t nbytes = (size_t)width * (size_t)heigth; //need to allocate the buffer of previous frame
    size_t nbytes_out = nbytes * 3;

    if(initialized == false){

        //at the first frame we can't compute the optical flow beacause this operation
        // require two frame ; so i use the first frame to do the necessary initialization
        // and show a black output

        CUDA_CHECK(cudaMalloc((void**)&buffer_previous_frame, nbytes));
        CUDA_CHECK(cudaMemcpy(buffer_previous_frame, buffer_current_frame, nbytes, cudaMemcpyDeviceToDevice));

        object_algorithm = cv::cuda::FarnebackOpticalFlow::create();

        CUDA_CHECK(cudaMemset(buffer_output_frame, 0, nbytes_out));

        initialized = true;
        return;

    }

    cv::cuda::GpuMat previous_frame(heigth, width, CV_8UC1, (void*)buffer_previous_frame, (size_t)width);
    cv::cuda::GpuMat current_frame(heigth, width, CV_8UC1, (void*)buffer_current_frame, (size_t)width);
    cv::cuda::GpuMat output_frame(heigth, width, CV_8UC3, (void*)buffer_output_frame, (size_t)width * 3);
    cv::cuda::GpuMat d_flow; //if i don't establish the parameters they are adattable 
    //d_flow will contain a 2d vector for each pixel of the frame (a frame with 2 channels)

    object_algorithm->calc(previous_frame, current_frame, d_flow);

    cv::cuda::GpuMat xy[2];
    cv::cuda::split(d_flow, xy);

    cv::cuda::GpuMat d_mag, d_ang;
    cv::cuda::cartToPolar(xy[0], xy[1], d_mag, d_ang, true);

    cv::cuda::GpuMat d_h;
    d_ang.convertTo(d_h, CV_8U, 0.5); //to bring tha angles between 0 and 180 degrees

    cv::cuda::GpuMat d_v;
    //cv::cuda::normalize(d_mag, d_v, 0, 255, cv::NORM_MINMAX, CV_8U);
    d_mag.convertTo(d_v, CV_8U, 12.0);

    cv::cuda::GpuMat d_s(d_h.size(), CV_8U);
    d_s.setTo(cv::Scalar(255));

    cv::cuda::GpuMat hsv;
    cv::cuda::GpuMat hsv_channels[3] = {d_h, d_s, d_v};
    cv::cuda::merge(hsv_channels, 3, hsv);

    //convert hsv to bgr
    cv::cuda::GpuMat bgr;
    //cv::cuda::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    hsv.copyTo(output_frame);

    CUDA_CHECK(cudaMemcpy(buffer_previous_frame, buffer_current_frame, nbytes, cudaMemcpyDeviceToDevice));


}