#include <cuda_runtime.h>
#include "cuda_common.h"

//aggiungere i parametri mancanti o eliminare quelli inutili
__global__ void optical_flow_new(unsigned char* prev, unsigned char* curr, float* output, int stride, int num_point_grid_x, int num_point_grid_y, int margin, int step){

    //patch 9x9
    //questo kernel è eseguito su un pixel/punto
    //dI_x e dI_y vanno calcolati sul frame precedente prev e non curr

    int index_thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    int index_thread_y = threadIdx.y + blockIdx.y * blockDim.y;
    if(index_thread_x >= num_point_grid_x || index_thread_y >= num_point_grid_y) return;

    int point_x = margin + (blockDim.x * blockIdx.x + threadIdx.x) * step;
    int point_y = margin + (blockDim.y * blockIdx.y + threadIdx.y) * step;

    int idx_a;
    int idx_b;
    int idx_c;
    int idx_e;
    int idx_d;

    int idx_output;

    float A = 0;
    float dI_x = 0;
    float B = 0;
    float dI_y = 0;
    float C = 0;
    float dI_time;
    float D = 0;
    float E = 0;

    float determinante ;  //per il calcolo della matrice inversa
    float matrice_inversa[2][2];

    float u;
    float v;

    int x;
    int y;

    for(x = -4; x <= 4; x++){
        for(y = -4; y <= 4; y++){
            idx_a = (point_y + y) * stride + (point_x + x);
            dI_x = ((float)(prev[idx_a + 1] - prev[idx_a - 1])) / 2;
            A += dI_x * dI_x;
        }
    }

    for(x = -4; x <= 4; x++){
        for(y = -4; y <= 4; y++){
            idx_b = (point_y + y) * stride + (point_x + x);
            dI_x = ((float)(prev[idx_b + 1] - prev[idx_b - 1])) / 2;
            dI_y = ((float)(prev[idx_b + stride] - prev[idx_b - stride])) / 2;
            B += dI_x * dI_y;
        }
    }

    for(x = -4; x <= 4; x++){
        for(y = -4; y <= 4; y++){
            idx_c = (point_y + y) * stride + (point_x + x);
            dI_y = ((float)(prev[idx_c + stride] - prev[idx_c - stride])) / 2;
            C += dI_y * dI_y;
        }
    }


    //adesso calcolare il determinante 
    determinante = A * C - B * B;
    if(determinante == 0) return;


    //calcolo matrice inversa
    matrice_inversa[0][0] = C / determinante;
    matrice_inversa[0][1] = -B / determinante;
    matrice_inversa[1][0] = -B / determinante;
    matrice_inversa[1][1] = A / determinante;

    
    //calcolo D,E

    for(x = -4; x <= 4; x++){
        for(y = -4; y <= 4; y++){
            idx_d = (point_y + y) * stride + (point_x + x);
            dI_x = ((float)(prev[idx_d + 1] - prev[idx_d - 1])) / 2;
            dI_time = curr[idx_d] - prev[idx_d];
            D += dI_x * dI_time;
        }
    }

    for(x = -4; x <= 4; x++){
        for(y = -4; y <= 4; y++){
            idx_e = (point_y + y) * stride + (point_x + x);
            dI_y = ((float)(prev[idx_e + stride] - prev[idx_e - stride])) / 2;
            dI_time = curr[idx_e] - prev[idx_e];
            E += dI_y * dI_time;
        }
    }

    //calcolo le coordinate del vettore di movimento del pixel in questione
    u = matrice_inversa[0][0] * (-D) + matrice_inversa[0][1] * (-E);
    v = matrice_inversa[1][0] * (-D) + matrice_inversa[1][1] * (-E);

    //immagino che l'ouput generico dell'optical flow sia una matrice di float dove inserire u e v per ciascun punto della 
    //griglia elaborato 

    idx_output = (index_thread_y * (blockDim.x * gridDim.x) + index_thread_x) * 2; // *2 in quanro l'output è float
    ouput[idx_output] = u;
    ouput[idx_output + 1] = v;
    

}


void run_optical_flow_new(bool* first_image, unsigned char* curr, unsigned char* prev, float* output, size_t nbytes, int num_point_grid_x, int num_point_grid_y){

    //caso in cui si elabora il primo frame 
    if(*first_image == true){
        CUDA_CHECK(cudaMemcpy(prev, curr, nbytes, cudaMemcpyDeviceToDevice));
        *first_image = false;
        return;
    }

    //caso standard , in cui si può calcolare l'optical flow avendo due frame
    dim3 block(16, 16);
    dim3 grid( (num_point_grid_x + block.x - 1) / block.x, (num_point_grid_y + block.y - 1) / block.y);

    optical_flow_new<<<grid, block>>>(prev, curr, output, stride, num_point_grid_x, num_point_grid_y, margin, step);

    CUDA_CHECK(cudaMemcpy(prev, curr, nbytes, cudaMemcpyDeviceToDevice));


}