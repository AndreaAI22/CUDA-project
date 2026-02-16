#include <cuda_runtime.h>
#include "cuda_common.h"


__global__ void optical_flow_new(unsigned char* prev, unsigned char* curr, float* output, int stride, int margin, int step, int num_point_grid_x, int num_point_grid_y){

    float tao = 0.001f;
    int idx, idx_output;
    float dI_x, dI_y, dI_time, u, v, determinante;
    float A = 0, B = 0, C = 0, D = 0, E = 0;
    float matrice_inversa[4];

    int index_thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_thread_y = threadIdx.y + blockIdx.y * blockDim.y;
    if(index_thread_x >= num_point_grid_x || index_thread_y >= num_point_grid_y) return;

    int point_x = margin + (blockDim.x * blockIdx.x + threadIdx.x) * step; //coordinata x del punto della griglia che si stà elaborando nel frame
    int point_y = margin + (blockDim.y * blockIdx.y + threadIdx.y) * step; //coordinata y del punto della griglia che si stà elaborando nel frame

    //patch/finestra di elaborazione 21x21
    for(int x = -10; x <= 10; x++){
        for(int y = -10; y <= 10; y++){

            idx = (point_y + y) * stride + (point_x + x);
            dI_x = ((float)(prev[idx + 1] - prev[idx - 1])) / 2;
            dI_y = ((float)(prev[idx + stride] - prev[idx - stride])) / 2;
            dI_time = (float)curr[idx] - prev[idx];
            A += dI_x * dI_x;
            B += dI_x * dI_y;
            C += dI_y * dI_y;
            D += dI_x * dI_time;
            E += dI_y * dI_time;

        }
    }

    idx_output = (index_thread_y * num_point_grid_x + index_thread_x) * 2; // *2 perchè per ogni punto della griglia abbiamo 2 componenti

    determinante = A * C - B * B;
    if(determinante < tao * (A + C) * (A + C)){
        output[idx_output] = 0;
        output[idx_output + 1] = 0;
        return;
    }

    //calcolo matrice inversa
    matrice_inversa[0] = C / determinante;
    matrice_inversa[1] = -B / determinante;
    matrice_inversa[2] = -B / determinante;
    matrice_inversa[3] = A / determinante;

    //calcolo le componenti del vettore di movimento del punto della griglia in questione
    u = matrice_inversa[0] * (-D) + matrice_inversa[1] * (-E);
    v = matrice_inversa[2] * (-D) + matrice_inversa[3] * (-E);

    output[idx_output] = u;
    output[idx_output + 1] = v;
    
}


void run_optical_flow_new(bool* first_image_bool, unsigned char* curr, unsigned char* prev, float* output, size_t nbytes, int num_point_grid_x, int num_point_grid_y, int margin, int step, int stride){

    //caso in cui si elabora il primo frame 
    if(*first_image_bool == true){
        CUDA_CHECK(cudaMemcpy(prev, curr, nbytes, cudaMemcpyDeviceToDevice));
        *first_image_bool = false;
        return;
    }

    //caso standard , in cui si può calcolare l'optical flow avendo a disposizione 2 frame
    dim3 block(16, 16);
    dim3 grid( (num_point_grid_x + block.x - 1) / block.x, (num_point_grid_y + block.y - 1) / block.y);

    optical_flow_new<<<grid, block>>>(prev, curr, output, stride, margin, step, num_point_grid_x, num_point_grid_y);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(prev, curr, nbytes, cudaMemcpyDeviceToDevice));

}