#include "CudaKernels.cuh"
#include <cstdio>
#include <cmath>

// Define array in constant memory, to be read by CUDA kernels


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

    // CPU -> GPU
    cudaMemcpy(*d_paddedImg, paddedImg->data, paddedSize, cudaMemcpyHostToDevice);

    freeImage(paddedImg);
}

// GPU -> CPU and free GPU memory
static void copyBackAndFree(
    Image* output, uint8_t* d_paddedImg, uint8_t* d_output)
{
    size_t outputSize = (size_t)output->width * output->height * 3;
    cudaMemcpy(output->data, d_output, outputSize, cudaMemcpyDeviceToHost);
    cudaFree(d_paddedImg);
    cudaFree(d_output);
}


// input: padded img, output: output img (both on GPU)
// kernelData: coefficients in global memory
__global__ void kernel1ChNoConst( const uint8_t* input, uint8_t* output, const float* kernelData, int width, int height, int paddedWidth, int kernelSize) {

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
                acc += src[(y + ky) * paddedWidth + (x + kx)] * kernelData[ky * kernelSize + kx];
            }
        }
        dst[y * width + x] = clampToUint8(acc);
    }
}


// Returns GPU time, total time, and speedup vs CPU
BenchmarkResult launch1ChNoConst(const Image* input, Image* output, const Kernel& kernel, dim3 blockSize, float cpuMs){
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

    //  GPU -> CPU and free GPU memory
    copyBackAndFree(output, d_paddedImg, d_output);
    cudaFree(d_kernel);

    float totalMs = stopTimer(totalStart, totalStop);

    // Return BenchmarkResult, i.e. GPU time, total time and speedup (vs CPU)
    return {gpuMs, totalMs, cpuMs / gpuMs};
}

// __________________________________________________


// Analogous to 1chNoConst but kernel coefficients are stored in constant memory
__global__ void kernel1ChConst(const uint8_t* input, uint8_t* output, int width, int height, int paddedWidth, int kernelSize){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int srcPlane = paddedWidth * (height + kernelSize - 1);
    int dstPlane = width * height;

    for (int c = 0; c < 3; c++) {
        const uint8_t* src = input + c * srcPlane;
        uint8_t* dst = output + c * dstPlane;

        float acc = 0.f;
        for (int ky = 0; ky < kernelSize; ky++) {
            for (int kx = 0; kx < kernelSize; kx++) {
                // Read from const memory 
                acc += src[(y + ky) * paddedWidth + (x + kx)] * d_constKernel[ky * kernelSize + kx];
            }
        }
        dst[y * width + x] = clampToUint8(acc);
    }
}

// analogous to launch1ChNoConst but copies data in const memory instead of allocating in global memory and passing pointer to kernel
BenchmarkResult launch1ChConst(const Image* input, Image* output, const Kernel& kernel, dim3 blockSize, float cpuMs){

    uint8_t *d_paddedImg, *d_output;
    int paddedWidth, paddedHeight;

    cudaEvent_t totalStart, totalStop;
    startTimer(totalStart, totalStop);

    // Copy coeffs in const memory
    cudaMemcpyToSymbol(d_constKernel, kernel.data, kernel.size * kernel.size * sizeof(float));

    allocAndTransfer(input, kernel, &d_paddedImg, &d_output, &paddedWidth, &paddedHeight);

    dim3 gridSize(
        (input->width  + blockSize.x - 1) / blockSize.x,
        (input->height + blockSize.y - 1) / blockSize.y,
        1
    );

    cudaEvent_t kernelStart, kernelStop;
    startTimer(kernelStart, kernelStop);

    kernel1ChConst<<<gridSize, blockSize>>>(d_paddedImg, d_output, input->width, input->height, paddedWidth, kernel.size);

    float gpuMs = stopTimer(kernelStart, kernelStop);
    copyBackAndFree(output, d_paddedImg, d_output);
    float totalMs = stopTimer(totalStart, totalStop);

    return {gpuMs, totalMs, cpuMs / gpuMs};
}

// __________________________________________________


// Analogous but uses a 3d grid where the z dimension corresponds to the channel. 
// The 3 channels are processed in parallel instead of sequentially and each thread handles one pixel of one channel.
__global__ void kernel3ChGrid(const uint8_t* input, uint8_t* output, int width, int height, int paddedWidth, int kernelSize){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x >= width || y >= height) return;

    int srcPlaneDim = paddedWidth * (height + kernelSize - 1);
    int dstPlaneDim = width * height;

    const uint8_t* src = input  + c * srcPlaneDim;
    uint8_t* dst = output + c * dstPlaneDim;

    float acc = 0.f;
    for (int ky = 0; ky < kernelSize; ky++) {
        for (int kx = 0; kx < kernelSize; kx++) {
            acc += src[(y + ky) * paddedWidth + (x + kx)] * d_constKernel[ky * kernelSize + kx];
        }
    }
    dst[y * width + x] = clampToUint8(acc);
}

BenchmarkResult launch3ChGrid(const Image* input, Image* output, const Kernel& kernel, dim3 blockSize, float cpuMs){
    uint8_t *d_paddedImg, *d_output;
    int paddedWidth, paddedHeight;

    cudaEvent_t totalStart, totalStop;
    startTimer(totalStart, totalStop);

    cudaMemcpyToSymbol(d_constKernel, kernel.data, kernel.size * kernel.size * sizeof(float));

    allocAndTransfer(input, kernel, &d_paddedImg, &d_output, &paddedWidth, &paddedHeight);

    // z = 3 for 3 channels
    dim3 gridSize(
        (input->width  + blockSize.x - 1) / blockSize.x,
        (input->height + blockSize.y - 1) / blockSize.y,
        3
    );

    cudaEvent_t kernelStart, kernelStop;
    startTimer(kernelStart, kernelStop);

    kernel3ChGrid<<<gridSize, blockSize>>>(d_paddedImg, d_output, input->width, input->height, paddedWidth, kernel.size);

    float gpuMs = stopTimer(kernelStart, kernelStop);
    copyBackAndFree(output, d_paddedImg, d_output);
    float totalMs = stopTimer(totalStart, totalStop);

    return {gpuMs, totalMs, cpuMs / gpuMs};
}


// __________________________________________________

// Analogous but with a 2d grid
// Each thread processes the 3 channels of one pixel in parallel using separate accumulators (ILP)
__global__ void kernel3ChNoGrid(const uint8_t* input, uint8_t* output, int width, int height, int paddedWidth, int kernelSize){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int srcPlane = paddedWidth * (height + kernelSize - 1);
    int dstPlane = width * height;

    // 3 accumulators and pointers for the 3 channels, allocated in registers
    float accR = 0.f, accG = 0.f, accB = 0.f;

    const uint8_t* srcR = input + 0 * srcPlane;
    const uint8_t* srcG = input + 1 * srcPlane;
    const uint8_t* srcB = input + 2 * srcPlane;

    for (int ky = 0; ky < kernelSize; ky++) {
        for (int kx = 0; kx < kernelSize; kx++) {
            float w = d_constKernel[ky * kernelSize + kx];
            int idx = (y + ky) * paddedWidth + (x + kx);
            // multiplications are independent and can be executed in parallel
            accR += srcR[idx] * w;
            accG += srcG[idx] * w;
            accB += srcB[idx] * w;
        }
    }

    output[0 * dstPlane + y * width + x] = clampToUint8(accR);
    output[1 * dstPlane + y * width + x] = clampToUint8(accG);
    output[2 * dstPlane + y * width + x] = clampToUint8(accB);
}

BenchmarkResult launch3ChNoGrid(const Image* input, Image* output, const Kernel& kernel, dim3 blockSize, float cpuMs){
    uint8_t *d_paddedImg, *d_output;
    int paddedWidth, paddedHeight;

    cudaEvent_t totalStart, totalStop;
    startTimer(totalStart, totalStop);

    cudaMemcpyToSymbol(d_constKernel, kernel.data, kernel.size * kernel.size * sizeof(float));

    allocAndTransfer(input, kernel, &d_paddedImg, &d_output, &paddedWidth, &paddedHeight);

    dim3 gridSize(
        (input->width  + blockSize.x - 1) / blockSize.x,
        (input->height + blockSize.y - 1) / blockSize.y,
        1
    );

    cudaEvent_t kernelStart, kernelStop;
    startTimer(kernelStart, kernelStop);

    kernel3ChNoGrid<<<gridSize, blockSize>>>(d_paddedImg, d_output, input->width, input->height, paddedWidth, kernel.size);

    float gpuMs = stopTimer(kernelStart, kernelStop);
    copyBackAndFree(output, d_paddedImg, d_output);
    float totalMs = stopTimer(totalStart, totalStop);

    return {gpuMs, totalMs, cpuMs / gpuMs};
}

// __________________________________________________


// separate pointers for the 3 channels (planar layout) -> 6 pointers for i/o
__global__ void kernel3Ch3Arrays(const uint8_t* inputR, const uint8_t* inputG, const uint8_t* inputB, 
    uint8_t* outputR, uint8_t* outputG, uint8_t* outputB, int width, int height, int paddedWidth, int kernelSize){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float accR = 0.f, accG = 0.f, accB = 0.f;

    for (int ky = 0; ky < kernelSize; ky++) {
        for (int kx = 0; kx < kernelSize; kx++) {
            float w = d_constKernel[ky * kernelSize + kx];
            int idx = (y + ky) * paddedWidth + (x + kx);

            accR += inputR[idx] * w;
            accG += inputG[idx] * w;
            accB += inputB[idx] * w;
        }
    }

    int outIdx = y * width + x;
    outputR[outIdx] = clampToUint8(accR);
    outputG[outIdx] = clampToUint8(accG);
    outputB[outIdx] = clampToUint8(accB);
}

BenchmarkResult launch3Ch3Arrays(const Image* input, Image* output, const Kernel& kernel, dim3 blockSize, float cpuMs){
   
    cudaEvent_t totalStart, totalStop;
    startTimer(totalStart, totalStop);

    cudaMemcpyToSymbol(d_constKernel, kernel.data, kernel.size * kernel.size * sizeof(float));

    int padding = kernel.size / 2;
    Image* padded = addPadding(input, padding);

    int paddedWidth = padded->width;
    int paddedHeight = padded->height;
    size_t paddedSize  = (size_t)paddedWidth * paddedHeight; 
    size_t outputSize  = (size_t)input->width * input->height;

    uint8_t *d_R, *d_G, *d_B; 
    uint8_t *d_Rout, *d_Gout, *d_Bout; 

    cudaMalloc(&d_R, paddedSize); 
    cudaMalloc(&d_G, paddedSize); 
    cudaMalloc(&d_B, paddedSize);
    cudaMalloc(&d_Rout, outputSize); 
    cudaMalloc(&d_Gout, outputSize); 
    cudaMalloc(&d_Bout, outputSize);

    // Copy each plane separately
    cudaMemcpy(d_R, padded->data + 0 * paddedSize, paddedSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_G, padded->data + 1 * paddedSize, paddedSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, padded->data + 2 * paddedSize, paddedSize, cudaMemcpyHostToDevice);

    freeImage(padded);

    dim3 gridSize(
        (input->width  + blockSize.x - 1) / blockSize.x,
        (input->height + blockSize.y - 1) / blockSize.y,
        1
    );

    cudaEvent_t kernelStart, kernelStop;
    startTimer(kernelStart, kernelStop);

    kernel3Ch3Arrays<<<gridSize, blockSize>>>(
        d_R, d_G, d_B, d_Rout, d_Gout, d_Bout,
        input->width, input->height, paddedWidth, kernel.size
    );

    float gpuMs = stopTimer(kernelStart, kernelStop);

    cudaMemcpy(output->data + 0 * outputSize, d_Rout, outputSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(output->data + 1 * outputSize, d_Gout, outputSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(output->data + 2 * outputSize, d_Bout, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_R);
    cudaFree(d_G);
    cudaFree(d_B);
    cudaFree(d_Rout); 
    cudaFree(d_Gout); 
    cudaFree(d_Bout);

    float totalMs = stopTimer(totalStart, totalStop);

    return {gpuMs, totalMs, cpuMs / gpuMs};
}


// Validation 
bool validateResults(const Image* cpuOutput, const Image* gpuOutput) {
    int n = planeSize(cpuOutput) * 3;
    for (int i = 0; i < n; i++) {
        if (abs((int)cpuOutput->data[i] - (int)gpuOutput->data[i]) > 1) {
            printf("Mismatch found at index %d\n", i);
            return false;
        }
    }
    return true;
}