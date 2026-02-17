#include <cstddef>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include "cuda_common.h"

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

    //variable for scale operation
    int output_width = 0;
    int output_height = 0;
    float scale_factor_x = 0;
    float scale_factor_y = 0;

    //variables to establish the output frame dimension according to operation
    int current_width;
    int current_height;

    size_t nbytes_out = 0;

    int limit = 0; //threshold value for edge detection


    //variables for optical flow new
    bool initial_frame_bool = true;
    int num_point_grid_x, num_point_grid_y, step, margin;
    float* buff_out_optical_flow_host = nullptr;
    float* buff_out_optical_flow_device = nullptr;
    float* buff_out_host_copy = nullptr;
    size_t size_buff_out;
    cv::Point arrow;


    //get the input to decide which operations must be execute
    int code_operation = 0;
    printf("Select the operation: \n");
    printf("Edge Detection = 1\nCrop = 2\nScale = 3\nRotate = 4\nOptical Flow = 5\n -->  ");
    std::cin >> code_operation;

    //connection to the video source (in this case is webcam
    cv::VideoCapture cap("/home/andrea/cuda_project/città_inglese.mp4");
    if (cap.isOpened() == false){
        fprintf(stderr, "Error: the video source can't be open!\n");
        return 1;
    }

    //add to menage correctly the stream video from file video.mp4
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0 || fps != fps) fps = 30.0; // fallback se non disponibile
    int delay_ms = (int)(1000.0 / fps);

    
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
            printf("\nInsert width output image --> ");
            std::cin >> output_width;
            if(output_width < 1) output_width = 1; //force to 1
            if (output_width > 4096) output_width = 4096;
            printf("\nInsert height output image --> ");
            std::cin >> output_height;
            if(output_height < 1) output_height = 1; //force to 1
            if (output_height > 4096) output_height = 4096;

            scale_factor_x = (float)w / (float)output_width;
            scale_factor_y = (float)h / (float)output_height;
            current_width = output_width;
            current_height = output_height;
            nbytes_out = (size_t)current_width * (size_t)current_height ;
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

            step = 20;
            margin = 12;

            num_point_grid_x = floor((current_width - 2 * margin) / step) + 1;
            num_point_grid_y = floor((current_height - 2 * margin) / step) + 1;

            size_buff_out = (num_point_grid_x * num_point_grid_y) * 2; 
            //*2 perchè per ogni punto della griglia cacoliamo il vettore di movimento fatto da 2 compoenti

            //allocazione per output optical flow
            buff_out_optical_flow_host = (float*) malloc(sizeof(float) * size_buff_out);
            if(buff_out_optical_flow_host == nullptr){
                fprintf(stderr, "Error: malloc failed!\n");
                return 1;
            }

            CUDA_CHECK(cudaMalloc((void**)&buff_out_optical_flow_device, size_buff_out * sizeof(float)));

            //buffer copia di buff_out_optical_flow per mantenere il risultato dell'optical flow del frame precedente per visualizzare meglio i vettori
            //con inizializzazione  a 0
            buff_out_host_copy = (float*) malloc(sizeof(float) * size_buff_out);
            if(buff_out_host_copy == nullptr){
                fprintf(stderr, "Error: malloc failed!\n");
                return 1;
            }
            buff_out_host_copy = (float*) memset(buff_out_host_copy, 0, size_buff_out * sizeof(float));
            if(buff_out_host_copy == nullptr){
                fprintf(stderr, "Error: memset failed!\n");
                return 1;
            }


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
                run_scale_kernel(buffer_device_in, buffer_device_out, current_width, current_height, w, h, scale_factor_x, scale_factor_y);
                break;

            case 4:
                run_rotate_kernel(buffer_device_in, buffer_device_out, current_width, current_height, c, s);
                break;

            case 5:
                //importante: buffer_device_in = curr mentre buffer_device_out = prev
                run_optical_flow_new(&initial_frame_bool, buffer_device_in, buffer_device_out, buff_out_optical_flow_device, nbytes, num_point_grid_x, num_point_grid_y, margin, step, w);
                break;
            
            default:
                fprintf(stderr, "ERROR: Invalid operation!\n");
                quit = true;
                break;
            
        }

        if(quit == true)    break;

        //do the copy of the elaborated frame from DEVICE to HOST
        if(code_operation != 5){
            CUDA_CHECK(cudaMemcpy(buffer_host_out, buffer_device_out, nbytes_out, cudaMemcpyDeviceToHost));
        }else{ //optical flow 
            CUDA_CHECK(cudaMemcpy(buffer_host_out, buffer_device_in, nbytes_out, cudaMemcpyDeviceToHost)); //disegno le frecce sul frame curr
            CUDA_CHECK(cudaMemcpy(buff_out_optical_flow_host, buff_out_optical_flow_device, size_buff_out*sizeof(float), cudaMemcpyDeviceToHost));
        }

        cv::Mat frame_output;
        frame_output = cv::Mat(current_height, current_width, CV_8UC1, buffer_host_out);

        //per visualizzazione delle frecce indicative del movimento 
        if(code_operation == 5){

            int origin_x, origin_y, idx_out;
            float componente_x, componente_y, modulo_vettore_spostamento, arrowhead_x, arrowhead_y;
            float fattore_scala = 20.0f, min_modulo = 0.15f, max_modulo = 10.0f, alpha = 0.25f;
            cv::Point origin, arrowhead;

            for(int i=0; i < num_point_grid_x; i++){
                for(int j=0; j < num_point_grid_y; j++){
                    
                    //coordinate punto griglia corrente , origine della freccia indicativa del movimento
                    origin_x = margin + step * i;
                    origin_y = margin + step * j;
                    origin = cv::Point(origin_x, origin_y);

                    //compoente x e y del vettore spostamento
                    idx_out = (i + j * num_point_grid_x) * 2;
                    componente_x = alpha * buff_out_optical_flow_host[idx_out] + (1 - alpha) * buff_out_host_copy[idx_out];
                    componente_y = alpha * buff_out_optical_flow_host[idx_out + 1] + (1 - alpha) *buff_out_host_copy[idx_out + 1];
                    if(isfinite(componente_x) == false || isfinite(componente_y) == false) continue;

                    buff_out_host_copy[idx_out] = componente_x;
                    buff_out_host_copy[idx_out + 1] = componente_y;

                    modulo_vettore_spostamento = sqrt(componente_x * componente_x + componente_y * componente_y);
                    if(isfinite(modulo_vettore_spostamento) == false) continue;

                    //per eliminare vettori troppo piccoli che quasi sempre sono rumore o troppo grandi dovuta a stima sbagliata
                    if(modulo_vettore_spostamento < min_modulo || modulo_vettore_spostamento > max_modulo) continue;
            
                    //normalizzazione per avere frecce tutte della stessa dimensione, per una visualizzazione migliore
                    arrowhead_x = origin_x + (componente_x / modulo_vettore_spostamento) * fattore_scala;
                    arrowhead_y = origin_y + (componente_y / modulo_vettore_spostamento) * fattore_scala; 
                    arrowhead = cv::Point((int)arrowhead_x, (int)arrowhead_y);

                    cv::arrowedLine(frame_output, origin, arrowhead, cv::Scalar(255), 2);

                }
            }
        }

        //display to screen
        cv::imshow("Result ", frame_output);

        int key = cv::waitKey(delay_ms);

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

    if(code_operation == 5){
        free(buff_out_host_copy);
        free(buff_out_optical_flow_host);
        CUDA_CHECK(cudaFree(buff_out_optical_flow_device));
    }

    return 0;
}