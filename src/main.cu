#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cstring>
#include <sys/stat.h>

#include "ImageIO.h"
#include "Sequential.h"
#include "CudaKernels.cuh"


static const int N_ITERATIONS = 20;   
static const int N_DISCARD = 2; // to avoid cold cache effects
static const float SIGMA  = 1.0f;  // sigma for gaussian kernel

// Test images
struct Resolution {const char* path; const char* name;};
static const Resolution RESOLUTIONS[] = {
    {"images/360p.jpeg","360p"},
    {"images/720p.jpeg","720p"},
    {"images/1080p.jpeg","1080p"},
    {"images/2K.jpeg","2K"},
    {"images/4K.jpeg","4K"}
};
static const int N_RESOLUTIONS = 5;

// Kernel sizes (side lengths)
static const int KERNEL_SIZES[] = {3, 5, 11, 21, 33};
static const int N_KERNEL_SIZES = 5;

// Block sizes 
struct BlockDim {int x; int y;};
static const BlockDim BLOCK_SIZES[] = {{8,8}, {16,16}, {32,32}};
static const int N_BLOCK_SIZES = 3;

// 5 variants of the GPU implementation
static const char* VARIANT_NAMES[] = {
    "1ChNoConst",
    "1ChConst",
    "3ChGrid",
    "3ChNoGrid",
    "3Ch3Arrays"
};
static const int N_VARIANTS = 5;





// Launch the specified GPU variant and return its benchmark result
static BenchmarkResult launchVariant(int variant, const Image* input, Image* output, const Kernel& kernel, dim3 blockSize, float cpuMs) {
    switch (variant) {
        case 0: return launch1ChNoConst(input, output, kernel, blockSize, cpuMs);
        case 1: return launch1ChConst(input, output, kernel, blockSize, cpuMs);
        case 2: return launch3ChGrid(input, output, kernel, blockSize, cpuMs);
        case 3: return launch3ChNoGrid(input, output, kernel, blockSize, cpuMs);
        case 4: return launch3Ch3Arrays(input, output, kernel, blockSize, cpuMs);
        default:
            std::cerr << "Invalid variant index: " << variant << std::endl;
            return {0.f, 0.f, 0.f};
    }
}





// Validate GPU results against CPU reference output
static void runValidation() {
    Image* input = loadImage(RESOLUTIONS[0].path);
    if (!input) {
        std::cerr << "Failed to load image for validation: " << RESOLUTIONS[0].path << std::endl;
        return;
    }
    Image* outputCPU = allocImage(input->width, input->height);
    Image* outputGPU = allocImage(input->width, input->height);
    Kernel kernel = makeKernel(KernelType::GAUSSIAN, 11, SIGMA); 
    sequentialConvolution(input, outputCPU, kernel);
    
    dim3 blockSize(16, 16, 1);
    bool allPassed = true;

    // Run all GPU variants and compare results
    for (int v = 0; v < N_VARIANTS; v++) {
        std::cout << "Validating variant: " << VARIANT_NAMES[v] << std::endl;
        
        std::fill(outputGPU->data, outputGPU->data + input->width * input->height * 3, 0);
        launchVariant(v, input, outputGPU, kernel, blockSize, 0.f);
        
        bool passed = validateResults(outputCPU, outputGPU);
        std::cout << "Validation " << (passed ? "PASSED" : "FAILED") << std::endl;
        if (!passed) allPassed = false;
    }
    std::cout << "Overall result: " << (allPassed ? "PASSED" : "FAILED") << std::endl;

    freeKernel(kernel);
    freeImage(input);
    freeImage(outputCPU);
    freeImage(outputGPU);
}



static float measureCPUTime(const Image* input, Image* output, const Kernel& kernel) {
    auto start = std::chrono::high_resolution_clock::now();
    sequentialConvolution(input, output, kernel);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}




// Benchmark all GPU variants across all configurations and save results to CSV
static void runBenchmark(std::ofstream& csv, const char* imgPath, const char* resolution, int kernelSize) {
    Image* input = loadImage(imgPath);
    if (!input) {
        std::cerr << "Failed to load image for benchmarking: " << imgPath << std::endl;
        return;
    }
    Image* outputCPU = allocImage(input->width, input->height);
    Image* outputGPU = allocImage(input->width, input->height);
    Kernel kernel = makeKernel(KernelType::GAUSSIAN, kernelSize, SIGMA); 
    

    // Measure CPU ----

    // Execute N_ITERATIONS runs
    // Discard the first N_DISCARD runs to mitigate cold cache effects
    float cpuMs[N_ITERATIONS];
    for (int i = 0; i < N_ITERATIONS; i++) {
        cpuMs[i] = measureCPUTime(input, outputCPU, kernel);        
    }

    float cpuMsSum = 0.f;
    for (int i = N_DISCARD; i < N_ITERATIONS; i++) {
        cpuMsSum += cpuMs[i];
    }
    float cpuAvgMs = cpuMsSum / (N_ITERATIONS - N_DISCARD);

    // Write CPU results to CSV
     csv << resolution << ","
        << input->width << "," << input->height << ","
        << kernelSize << ","
        << "CPU,N/A,N/A,"
        << cpuAvgMs << "," << cpuAvgMs << ",1.0\n";

    std::cout << "[" << resolution << " k=" << kernelSize << "] "
              << "CPU: " << cpuAvgMs << " ms\n";


    // Measure GPU ----

    // Execute N_ITERATIONS runs for every GPU variant and every block size
    // Discard the first N_DISCARD runs to mitigate cold cache effects 
    
    for(int v = 0; v < N_VARIANTS; v++) {
        for(int b = 0; b < N_BLOCK_SIZES; b++){
            dim3 blockSize(BLOCK_SIZES[b].x, BLOCK_SIZES[b].y, 1);
            float gpuMs[N_ITERATIONS];
            float totalMs[N_ITERATIONS];
            for (int i = 0; i < N_ITERATIONS; i++) {
                BenchmarkResult result = launchVariant(v, input, outputGPU, kernel, blockSize, cpuAvgMs);
                gpuMs[i] = result.gpuMs;
                totalMs[i] = result.totalMs;
            }

            float gpuMsSum = 0.f;
            float totalMsSum = 0.f;
            for (int i = N_DISCARD; i < N_ITERATIONS; i++) {
                gpuMsSum += gpuMs[i];
                totalMsSum += totalMs[i];
            }
            int nValidRuns = N_ITERATIONS - N_DISCARD;
            float gpuAvgMs = gpuMsSum / nValidRuns;
            float totalAvgMs = totalMsSum / nValidRuns;
            float speedup = cpuAvgMs / gpuAvgMs;

            // Write GPU results to CSV
            csv << resolution << ","
                << input->width << "," << input->height << ","
                << kernelSize << ","
                << VARIANT_NAMES[v] << ","
                << BLOCK_SIZES[b].x << "," << BLOCK_SIZES[b].y << ","
                << gpuAvgMs << "," << totalAvgMs << ","
                << speedup << "\n";

            std::cout << "  " << VARIANT_NAMES[v]
                    << " block=" << BLOCK_SIZES[b].x
                    << "x" << BLOCK_SIZES[b].y
                    << " gpuMs=" << gpuAvgMs
                    << " speedup=" << speedup << "x\n";
                
        }
    }    
            
}







int main() {
    // Run validation first
    runValidation();

    // Create results directory if it doesn't exist
    mkdir("results", 0755);


    // Experiment 1: fixed kernel size (11x11), varying img resolution
    std::ofstream csv("results/fixed_kernel_results.csv");
    csv << "resolution,imgWidth,imgHeight,kernelSize,variant,"
               "blockX,blockY,gpuMs,totalMs,speedup\n";

    for (int r = 0; r < N_RESOLUTIONS; r++) {
            runBenchmark(csv,
                RESOLUTIONS[r].path,
                RESOLUTIONS[r].name,
                11);
    }
    csv.close();
    std::cout << "Saved: results/fixed_kernel_results.csv\n";



    // Experiment 2: fixed img resolution (1080p), varying kernel size
    std::ofstream csv("results/fixed_resolution_results.csv");
    csv << "resolution,imgWidth,imgHeight,kernelSize,variant,"
               "blockX,blockY,gpuMs,totalMs,speedup\n";

    for (int ks = 0; ks < N_KERNEL_SIZES; ks++) {
        runBenchmark(csv,
            RESOLUTIONS[0].path,
            RESOLUTIONS[0].name,
            KERNEL_SIZES[ks]);
    }
    csv.close();
    std::cout << "Saved: results/fixed_resolution_results.csv\n";

    std::cout << "\nBenchmarking completed.\n";
    return 0;

}