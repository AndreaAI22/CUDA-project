#include <cuda_runtime.h>
#include "cuda_common.h"

//aggiungere i parametri mancanti o eliminare quelli inutili
__global__ void optical_flow_new(unsigned char* prev, unsigned char* curr, int stride, int point_x, int point_y){

    //patch 9x9
    //questo kernel è eseguito su un pixel/punto
    int idx_a;
    int idx_b;
    int idx_c;

    float A = 0;
    float dI_x = 0;
    float B = 0;
    float dI_y = 0;
    float C = 0;

    float determinante ;  //per il calcolo della matrice inversa
    float matrice_inversa[4];

    for(x = -4; x <= 4; x++){
        for(y = -4; y <= 4; y++){
            idx_a = (point_y + y) * stride + (point_x + x);
            dI_x = ((float)(curr[idx_a + 1] - curr[idx_a - 1])) / 2;
            A += dI_x * dI_x;
        }
    }

    for(x = -4; x <= 4; x++){
        for(y = -4; y <= 4; y++){
            idx_b = (point_y + y) * stride + (point_x + x);
            dI_x = ((float)(curr[idx_b + 1] - curr[idx_b - 1])) / 2;
            dI_y = ((float)(curr[idx_b + stride] - curr[idx_b - stride])) / 2;
            B += dI_x * dI_y;
        }
    }

    for(x = -4; x <= 4; x++){
        for(y = -4; y <= 4; y++){
            idx_c = (point_y + y) * stride + (point_x + x);
            dI_y = ((float)(curr[idx_c + stride] - curr[idx_c - stride])) / 2;
            C += dI_y * dI_y;
        }
    }


    //adesso calcolare il determinante 
    determinante = A * C - B * B;
    if(determinante == 0){
        fprintf(stderr, "la matrice non è invertibile");
        return;
    }

    //calcolo matrice inversa
    matrice_inversa[0] = C / determinante;
    matrice_inversa[1] = -B / determinante;
    matrice_inversa[2] = -B / determinante;
    matrice_inversa[3] = A / determinante;



}