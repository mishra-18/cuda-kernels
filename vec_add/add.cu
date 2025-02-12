#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 100000

__global__ void vec_add(float *out, float *a, float *b){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < N){
        out[i] = a[i] + b[i];
    }
}

int main(){

    float *out, *a, *b;
    float *d_o, *d_a, *d_b;

    // allocate host memory
    a = (float*)malloc(sizeof(float) * N);    
    b = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // initialize host memory
    for(int i=0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2*a[i];
    }

    // allocate device memory
    cudaMalloc((void**)&d_a, sizeof(float)*N);
    cudaMalloc((void**)&d_b, sizeof(float)*N);
    cudaMalloc((void**)&d_o, sizeof(float)*N);

    // transfer data from host to device
    cudaMemcpy(d_a, a, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*N, cudaMemcpyHostToDevice);
    
    // execute kernel
    int threadsPerBlock = 256;
    int nBlocks = (N + threadsPerBlock)/threadsPerBlock; 
    vec_add<<<nBlocks, threadsPerBlock>>>(d_o, d_a, d_b);

    // collect data from device to host
    cudaMemcpy(out, d_o, sizeof(float)*N, cudaMemcpyDeviceToHost);
    
    for(int i =0; i < 10; i++){
        printf("%f ", out[i]);
    }
    return 0;
}
