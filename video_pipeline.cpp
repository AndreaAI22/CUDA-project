#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include "sobel.h"

int main(){

    bool quit = false;
    const float pi = 3.14159265358979323846f;


    //variables for crop operation
    int roi_x = 0;
    int roi_y = 0;
    int roi_width = 0;
    int roi_height = 0;

    //variable for rotate operation
    float rotate_angle = 0;
    float c = 1.0f;
    float s = 0.0f;
    float theta = 0.0f;

    //variables to establish the output frame dimension according to operation
    int current_width;
    int current_height;

    size_t nbytes_out = 0;

    int limit = 0; //threshold value for edge detection


    //get the input to decide which operations must be execute
    int code_operation = 0;
    printf("Select the operation: \n");
    printf("Edge Detection = 1\nCrop = 2\nScale = 3\nRotate = 4\nOptical Flow = 5\n -->  ");
    std::cin >> code_operation;

    //connection to the video source (in this case is webcam)
    cv::VideoCapture cap(0);
    if (cap.isOpened() == false){
        fprintf(stderr, "Error: the video source can't be open!\n");
        return 1;
    }

    
    cv::Mat frame_original, frame_grayscale;

    //read a single frame to know the resolution and other properties

    cap >> frame_original;
    if(frame_original.empty()){
        fprintf(stderr, "Error: the frame can't be readen!\n");
        return 1;
    }

    //turn the frame in grayscale to make easier the computation
    cv::cvtColor(frame_original, frame_grayscale, cv::COLOR_BGR2GRAY);

    //check if the buffer is continuous (if there is padding)
    if(frame_grayscale.isContinuous() == false){
        frame_grayscale = frame_grayscale.clone();
    }

    int w = frame_grayscale.cols; //width of input frame
    int h = frame_grayscale.rows;  //height of input frame
    int channels = frame_grayscale.channels();
    size_t nbytes = (size_t)w * (size_t)h * (size_t)channels;
    
    //in this context where the format pixel is grayscale , 1 pixel(element) = 1 byte
    size_t n_elements = nbytes;

    switch( code_operation ){

        case 1: //the user chooses Edge Detection
            limit = 50;
            current_width = w;
            current_height = h;
            nbytes_out = nbytes;
            printf("Press 'u'(up) or 'd'(down) to change the threshold value (after click on the GUI)\n");
            break;

        case 2: //the user chooses Crop
            printf("\nInsert x-coordinate of ROI origin --> ");
            std::cin >> roi_x;
            printf("\nInsert y-coordinate of ROI origin --> ");
            std::cin >> roi_y;
            printf("\nInsert width of ROI --> ");
            std::cin >> roi_width;
            printf("\nInsert heigth of ROI --> ");
            std::cin >> roi_height;

            if(roi_x < 0) roi_x = 0; //if the user put a negative number for roi_x we put that to 0
            if(roi_y < 0) roi_y = 0; //same for roi_y
            if(roi_x >= w) roi_x = w-1; //if the user put a number too large for roi_x we put that to w-1
            if(roi_y >= h) roi_y = h-1; //same for roi_y
            if(roi_width < 1) roi_width = 1; //if the user put 0 or negative number for roi_width we put that to 1
            if(roi_height < 1) roi_height = 1; //same for roi_height

            if(roi_x + roi_width > w){
                printf("\nROI out of bounds; we have make smaller the ROI\n");
                roi_width = w - roi_x;
            }

            if(roi_y + roi_height > h){
                printf("\nROI out of bounds; we have make smaller the ROI\n");
                roi_height = h - roi_y;
            }

            //i put all these control on the user input to force the ROI inside the image

            current_width = roi_width;
            current_height = roi_height;
            nbytes_out = (size_t)current_width * (size_t)current_height;
            break;

        case 3: //the user chooses Scale
            current_width = w/3;
            current_height = h/3;
            nbytes_out = current_width * current_height ;
            break;

        case 4: //the user chooses Rotate
            printf("\nInsert rotate angle --> ");
            std::cin >> rotate_angle;

            //convert grades to radius
            theta = (rotate_angle * pi) / 180.0f;

            //compute here the cos and sin value because is more efficent than do it for each pixel in the kernel
            //i use the - to apply the original formula,and not the inversa in the kernel(inverse mapping)
            c = cosf(-theta);
            s = sinf(-theta);

            current_width = w;
            current_height = h;
            nbytes_out = nbytes;
            break;

        case 5: //the user chooses Optical Flow
            current_width = w;
            current_height = h;
            nbytes_out = nbytes;
            break;

        default: //invalid operation
            fprintf(stderr, "ERROR: Invalid OPeration\n");
            return 1;
            break;
        
    }


    unsigned char* buffer_host_in = frame_grayscale.data;

    //allocate buffer host for output and the two buffer in device(GPU)
    unsigned char* buffer_host_out = (unsigned char*) malloc(nbytes_out);
    if(buffer_host_out == nullptr){
        fprintf(stderr, "Error: malloc failed!\n");
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

        switch( code_operation ){

            case 1:
                run_edge_detection_sobel(buffer_device_in, buffer_device_out, w, h, limit);
                break;

            case 2:
                run_crop_kernel(buffer_device_in, buffer_device_out, roi_x, roi_y, roi_width, roi_height, w, h);
                break;

            case 3:
                run_scale_kernel(buffer_device_in, buffer_device_out, current_width, current_height, w, h);
                break;

            case 4:
                run_rotate_kernel(buffer_device_in, buffer_device_out, current_width, current_height, c, s);
                break;

            case 5:
                run_optical_flow(buffer_device_in, buffer_device_out, current_width, current_height);
                break;
            
            default:
                fprintf(stderr, "ERROR: Invalid operation!\n");
                quit = true;
                break;
            
        }

        if(quit == true)    break;

        //do the copy of the elaborated frame from DEVICE to HOST
        CUDA_CHECK(cudaMemcpy(buffer_host_out, buffer_device_out, nbytes_out, cudaMemcpyDeviceToHost));

        cv::Mat frame_output(current_height, current_width, CV_8UC1, buffer_host_out);

        //display to screen
        cv::imshow("Result ", frame_output);

        int key = cv::waitKey(1);

        //if the user press Esc then stop
        if(key == 27){
            break;
        }
        
        //to change the threshold value of edge detection during the execution
        if(code_operation == 1 ){
            if(key == 'u'){
                if(limit < 250) limit += 5;
                else limit = 255;
            }
            if(key == 'd'){
                if(limit > 5) limit -= 5;
                else limit = 0;
            }
        }

    }

    CUDA_CHECK(cudaFree(buffer_device_in));
    CUDA_CHECK(cudaFree(buffer_device_out));
    free(buffer_host_out);

    return 0;
}