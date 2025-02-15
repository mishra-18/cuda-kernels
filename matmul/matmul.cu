#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WIDTH 64

__global__ void MatrixMult(float * dN, float * dM, float * dP, int width){ 
    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.y*blockIdx.x;

    if (row < width && col < width){
        float Pvalue = 0;
        for(int k=0; k < width; k++){
            Pvalue += dN[row*width + k]*dM[k*width + col];
        }

        dP[row*width + col] = Pvalue;
    }
    
}
int main(){
    float *N, *P, *M;
    float *dN, *dP, *dM;

    // Allocate host memory
    N = (float*)malloc(sizeof(float) * WIDTH*WIDTH); 
    M = (float*)malloc(sizeof(float) * WIDTH*WIDTH);    
    P = (float*)malloc(sizeof(float) * WIDTH*WIDTH);      

    // Initialize host memory
    for(int i=0; i < WIDTH*WIDTH; i++){
        N[i] = (float)rand()/RAND_MAX;
        M[i] = (float)rand()/RAND_MAX;            
    }

    // // Allocate device memory
    cudaMalloc((void**)&dN, sizeof(float)*WIDTH*WIDTH);
    cudaMalloc((void**)&dM, sizeof(float)*WIDTH*WIDTH);
    cudaMalloc((void**)&dP, sizeof(float)*WIDTH*WIDTH);

    // Move from host to device
    cudaMemcpy(dN, N, sizeof(float)*WIDTH*WIDTH, cudaMemcpyHostToDevice);
    cudaMemcpy(dM, M, sizeof(float)*WIDTH*WIDTH, cudaMemcpyHostToDevice);

    // Execute the kernel
    int TILE_SIZE = 16;
    dim3 gridsDim(ceil(WIDTH/TILE_SIZE), ceil(WIDTH/TILE_SIZE));
    dim3 threadsDim(TILE_SIZE, TILE_SIZE);
    MatrixMult<<<gridsDim, threadsDim>>>(dN, dM, dP, WIDTH);

    // Load it back to host
    cudaMemcpy(P, dP, sizeof(float)*WIDTH*WIDTH, cudaMemcpyDeviceToHost);
    
    cudaFree(dP); cudaFree(dN), cudaFree(dM);
    free(N), free(M), free(P);

    return 0;
}