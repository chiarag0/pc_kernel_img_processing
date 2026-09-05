#include "Sequential.h"
#include "ImageIO.h"
#include <cmath>
#include <cstring>
#include <algorithm>


Kernel makeKernel(KernelType type, int size, float sigma) {
    Kernel kernel;
    kernel.size = size;
    kernel.data = new float[size * size]();
 
    switch (type) {
        case KernelType::GAUSSIAN: {
            // Computes the coefficients of a Gaussian kernel of given size centered at the middle of the kernel
            float mean = size / 2.0f;
            float sum  = 0.0f;
 
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    float val = std::exp(
                        -0.5f * (std::pow((x - mean) / sigma, 2.0f) +
                                 std::pow((y - mean) / sigma, 2.0f))
                    ) / (2.0f * M_PI * sigma * sigma);
 
                    kernel.data[y * size + x] = val;
                    sum += val;
                }
            }
 
            // Normalisation so that the sum of all coefficients is 1.0 to preserve medium brightness
            for (int i = 0; i < size * size; i++)
                kernel.data[i] /= sum;
                
            break;
        }
    }
 
    return kernel;
}

void freeKernel(Kernel& kernel) {
    delete[] kernel.data;
    kernel.data = nullptr;
}



void sequentialConvolution(const Image* input, Image* output, const Kernel& kernel) {
    int padding   = kernel.size / 2;
    Image* paddedImg = addPadding(input, padding);
 
    int width = input->width;
    int height = input->height;
    int paddedWidth = paddedImg->width;
    int srcPlaneSize = planeSize(paddedImg);
    int dstPlaneSize = planeSize(output);
 

    // Convolution: for each output pixel, compute the weighted sum of the corresponding kernel-sized region in the padded input
    // Loop on channels 
    for (int c = 0; c < 3; c++) {
        const uint8_t* src = paddedImg->data + c * srcPlaneSize;
        uint8_t* dst = output->data + c * dstPlaneSize;
 
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float acc = 0.f;  // Accumulator for the convolution sum
                for (int ky = 0; ky < kernel.size; ky++) {
                    for (int kx = 0; kx < kernel.size; kx++) {
                        acc += src[(y + ky) * paddedWidth + (x + kx)]
                             * kernel.data[ky * kernel.size + kx];
                    }
                }
                dst[y * width + x] = static_cast<uint8_t>(
                    std::min(std::max(acc, 0.f), 255.f)
                );
            }
        }
    }
 
    freeImage(paddedImg);
}
