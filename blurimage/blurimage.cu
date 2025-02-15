#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cuda_runtime.h>

__global__ void imagetoBlur(unsigned char * pout, unsigned char * dpimg, int w, int h){
    int col = threadIdx.x + blockDim.x*blockIdx.x;
    int row = threadIdx.y + blockDim.y*blockIdx.y;

    int valSum = 0;
    int valCnt = 0;
    if (col < w && row < h){
        for(int i = row-1; i <=row+1; ++i){
            for(int j = col-1; j <=col+1; ++j){
                if(i >= 0 && j >= 0 && i<h && j<w){
                    int offset = i*w + j;
                    valSum += dpimg[offset];
                    ++valCnt;
                }
            }
        }
        pout[row*w + col] = (unsigned char)(valSum/valCnt);
    }   
}
int main(){
    int w, h, c;
    unsigned char *img = stbi_load("test.png", &w, &h, &c, 1);

    // Allocate to host
    unsigned char *pimg = (unsigned char *)malloc(sizeof(unsigned char)*w*h);
    unsigned char *blurimg = (unsigned char *)malloc(sizeof(unsigned char)*w*h);
    memcpy(pimg, img, sizeof(unsigned char)*w*h);

    // Allocate to device
    unsigned char *pout, *dpimg; 
    cudaMalloc((void**)&pout, sizeof(unsigned char)*w*h);
    cudaMalloc((void**)&dpimg, sizeof(unsigned char)*w*h);

    // Load from host to device 
    cudaMemcpy(dpimg, pimg, sizeof(unsigned char)*w*h, cudaMemcpyHostToDevice);
    
    // Execute the kernel
    int patchSize = 3; 
    dim3 blocksDim(ceil(w/patchSize), ceil(h/patchSize), 1);
    dim3 threadsDim(patchSize, patchSize, 1);
    imagetoBlur<<<blocksDim, threadsDim>>>(pout, dpimg, w, h);

    // Load to host
    cudaMemcpy(blurimg, pout, sizeof(unsigned char)*w*h, cudaMemcpyDeviceToHost);

    // save the blured image
    stbi_write_png("blurimg.png", w, h, 1, blurimg, w);

    return 0;
}