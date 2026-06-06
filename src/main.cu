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
    {"images/360p.png","360p"},
    {"images/720p.png","720p"},
    {"images/1080p.png","1080p"},
    {"images/4K.png","4K"}
};
static const int N_RESOLUTIONS = 4;

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

