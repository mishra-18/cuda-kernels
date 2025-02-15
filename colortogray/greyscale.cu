#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cuda_runtime.h>

__global__ void colortoGrayscale(unsigned char * Pout, unsigned char *dPimg, int width, int height){
   int col = threadIdx.x + blockDim.x*blockIdx.x;
   int row = threadIdx.y + blockDim.y*blockIdx.y;

   if (col < width && row < height){
    int grayoffset = row*width + col;

    int rgboffset = grayoffset*3;

    unsigned char r = dPimg[rgboffset];
    unsigned char g = dPimg[rgboffset + 1];
    unsigned char b = dPimg[rgboffset + 2];

    Pout[grayoffset] = 0.21f*r + 0.7f*g + 0.07f*b;
   }
}

int main(){
    int w, h, c;
    unsigned char *img = stbi_load("images.jpeg", &w, &h, &c, 3);

    // printf("%d, %d, %d", w, h, c);
    // Allocate vector to host
    unsigned char *Pimg = (unsigned char *)malloc(sizeof(unsigned char)*w * h * 3);
    unsigned char *grayscale = (unsigned char *)malloc(sizeof(unsigned char)*w*h);
    memcpy(Pimg, img, w*h*3);


    // Allocate vector to device
    unsigned char *Pout, *dPimg;
    cudaMalloc((void**)&Pout, sizeof(unsigned char)*w*h);
    cudaMalloc((void**)&dPimg, sizeof(unsigned char)*w*h*3);

    // Load from host to device
    cudaMemcpy(dPimg, Pimg, sizeof(unsigned char)*w*h*3, cudaMemcpyHostToDevice);

    // Execute the Kernel
    int numthreads = 16;
    int width = w;
    int row = h;

    dim3 blocksDim(ceil(width/numthreads), ceil(row/numthreads), 1);
    dim3 threadsDim(numthreads, numthreads, 1);
    colortoGrayscale<<<blocksDim, threadsDim>>>(Pout, dPimg, w, h);
    
    // Load back to host
    cudaMemcpy(grayscale, Pout, sizeof(unsigned char)*w*h, cudaMemcpyDeviceToHost);

    // Save it back and free up the space
    stbi_write_png("gray.png", w, h, 1, grayscale, w);
    
    return 0;
}