#pragma once
#include "ImageIO.h"
#include "Sequential.h"
 
#define MAX_KERNEL_DIM 64
 
// Kernel coefficients stored in constant memory.
constant float d_constKernel[MAX_KERNEL_DIM * MAX_KERNEL_DIM];
 
struct BenchmarkResult {
    float gpuMs;
    float totalMs; // gpu time + cudaMalloc + transfer times
    float speedup; // = cpuMs/gpuMs
}; 


// V1: global memory + loop on channels
BenchmarkResult launch1ChNoConst(const Image* input, Image* output, const Kernel& k, dim3 blockSize, float cpuMs);
 
// V2: constant memory + loop on channels
BenchmarkResult launch1ChConst(const Image* input, Image* output, const Kernel& k, dim3 blockSize, float cpuMs);
 
// V3: constant memory + 3d grid
BenchmarkResult launch3ChGrid(const Image* input, Image* output, const Kernel& k, dim3 blockSize, float cpuMs);
 
// V4: constant memory + 2d grid
BenchmarkResult launch3ChNoGrid(const Image* input, Image* output, const Kernel& k, dim3 blockSize, float cpuMs);
 
// V5: constant memory + planar layout with 3 pointers
BenchmarkResult launch3Ch3Arrays(const Image* input, Image* output, const Kernel& k, dim3 blockSize, float cpuMs);

bool validateResults(const Image* cpuOutput, const Image* gpuOutput);