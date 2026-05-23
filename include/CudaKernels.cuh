#pragma once
#include "ImageIO.h"
#include "Sequential.h"
 
#define MAX_KERNEL_DIM 64
 
// Kernel coefficients stored in constant memory.
extern __constant__ float d_constKernel[MAX_KERNEL_DIM * MAX_KERNEL_DIM];
 
struct BenchmarkResult {
    float gpuMs;  
    float totalMs; // gpu time + cudaMalloc + transfer times
    float speedup; // cpuMs / gpuMs
}; 