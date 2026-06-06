#pragma once
#include "ImageIO.h"

enum class KernelType {
    GAUSSIAN,
    SOBEL
};

struct Kernel {
    float* data;  // planar array: row 0 - row 1 - ... - row (size-1)
    int size; // kernel is size x size
};

Kernel makeKernel(KernelType type, int size = 3, float sigma = 1.0f);
void   freeKernel(Kernel& k);
 
void sequentialConvolution(const Image* input, Image* output, const Kernel& k);
void sequentialSobel(const Image* input, Image* output);