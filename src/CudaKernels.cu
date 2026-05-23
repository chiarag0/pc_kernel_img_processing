#include "CudaKernels.cuh"
#include <cstdio>
#include <cmath>

// Define array in constant memory, to be read by CUDA kernels
__constant__ float d_constKernel[MAX_KERNEL_DIM * MAX_KERNEL_DIM];


__device__ inline uint8_t clampToUint8(float val) {
    return (uint8_t)fminf(fmaxf(val, 0.f), 255.f);
}


// Initialize CUDA timer by creating the start and stop events and recording the start event
static void startTimer(cudaEvent_t& start, cudaEvent_t& stop) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
}

// Stop timer by recording the stop event, synchronizing it, calculating elapsed time in Ms and destroying the events
static float stopTimer(cudaEvent_t& start, cudaEvent_t& stop) {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}


// Helper function to allocate padded image on GPU and copy from CPU. Returns padded dimensions via pointers
static void allocAndTransfer( const Image* input, const Kernel& k, uint8_t** d_paddedImg, uint8_t** d_output,int* paddedWidth, int* paddedHeight){
    int padding = k.size / 2;
    Image* paddedImg = addPadding(input, padding);

    *paddedWidth = paddedImg->width;
    *paddedHeight = paddedImg->height;

    size_t paddedSize = (size_t)paddedImg->width * paddedImg->height * 3;
    size_t outputSize = (size_t)input->width  * input->height  * 3;

    cudaMalloc(d_paddedImg, paddedSize);
    cudaMalloc(d_output, outputSize);

    // CPU → GPU
    cudaMemcpy(*d_paddedImg, paddedImg->data, paddedSize, cudaMemcpyHostToDevice);

    freeImage(paddedImg);
}

// GPU → CPU and free GPU memory
static void copyBackAndFree(
    Image* output, uint8_t* d_paddedImg, uint8_t* d_output)
{
    size_t outputSize = (size_t)output->width * output->height * 3;
    cudaMemcpy(output->data, d_output, outputSize, cudaMemcpyDeviceToHost);
    cudaFree(d_paddedImg);
    cudaFree(d_output);
}



__global__ void kernel1ChNoConst(
    const uint8_t* input,   // padded image (GPU)
    uint8_t* output,        // output image (GPU)
    const float* kernelData,// kernel coefficients in global memory
    int width, int height,  // original image dimensions
    int paddedWidth,        // padded image width = width + 2*padding
    int kernelSize)         // kernel side length (3, 5, 11, ...)
{

    // Each thread computes the convolution for one pixel (x, y) of the image.
    // blockIdx = position of the block in the grid
    // blockDim = number of threads per block
    // threadIdx = position of the thread in the block
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Guard: if the thread is out of bounds of the image, exit immediately.
    if (x >= width || y >= height) return;

    int srcPlaneDim = paddedWidth * (height + kernelSize - 1); 
    int dstPlaneDim = width * height; 

    // Each thread processes the 3 channels sequentially
    for (int c = 0; c < 3; c++) {
        const uint8_t* src = input + c * srcPlaneDim;
        uint8_t* dst = output + c * dstPlaneDim;

        // Convolution sum for the pixel (x, y) in channel c
        // Identical to sequential (CPU) version
        float acc = 0.f;
        for (int ky = 0; ky < kernelSize; ky++) {
            for (int kx = 0; kx < kernelSize; kx++) {
                acc += src[(y + ky) * paddedWidth + (x + kx)]
                     * kernelData[ky * kernelSize + kx];
            }
        }
        dst[y * width + x] = clampToUint8(acc);
    }
}


// Returns GPU time, total time, and speedup vs CPU
BenchmarkResult launch1ChNoConst(
    const Image* input, Image* output,
    const Kernel& kernel, dim3 blockSize, float cpuMs)
{
    uint8_t *d_paddedImg, *d_output;
    int paddedWidth, paddedHeight;

    // Timer accounting for all GPU ops (memory allocation, data transfer, kernel execution, memory free)
    cudaEvent_t totalStart, totalStop;
    startTimer(totalStart, totalStop);

    // Allocate kernel coefficients in global memory 
    float* d_kernel;
    cudaMalloc(&d_kernel, kernel.size * kernel.size * sizeof(float));

    // Copy kernel coefficients from CPU to GPU global memory
    cudaMemcpy(d_kernel, kernel.data, kernel.size * kernel.size * sizeof(float), cudaMemcpyHostToDevice);

    allocAndTransfer(input, kernel, &d_paddedImg, &d_output, &paddedWidth, &paddedHeight);

    // Calculate grid size to cover the entire image with the given block size
    dim3 gridSize(
        (input->width  + blockSize.x - 1) / blockSize.x,
        (input->height + blockSize.y - 1) / blockSize.y,
        1
    );

    // Timer for kernel execution only
    cudaEvent_t kernelStart, kernelStop;
    startTimer(kernelStart, kernelStop);

    // Kernel launch specifying number of blocks (gridSize) and threads per block (blockSize) 
    kernel1ChNoConst<<<gridSize, blockSize>>>(d_paddedImg, d_output, d_kernel, input->width, input->height, paddedWidth, kernel.size);

    float gpuMs = stopTimer(kernelStart, kernelStop);

    //  GPU → CPU and free GPU memory
    copyBackAndFree(output, d_paddedImg, d_output);
    cudaFree(d_kernel);

    float totalMs = stopTimer(totalStart, totalStop);

    // Return BenchmarkResult, i.e. GPU time, total time and speedup (vs CPU)
    return {gpuMs, totalMs, cpuMs / gpuMs};
}

