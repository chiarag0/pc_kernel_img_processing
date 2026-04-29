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
 
        case KernelType::SOBEL: {
            // Sobel uses two separate kernels Gx and Gy managed in sequentialSobel()
            kernel.size = 3;
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


void sequentialSobel(const Image* input, Image* output) {
    int width = input->width;
    int height = input->height;
    int planeSize = width * height;
 
    // Gx and Gy are the standard 3x3 Sobel kernels for horizontal and vertical edge detection respectively
    float Gx[9] = { -1,  0,  1,
                    -2,  0,  2,
                    -1,  0,  1 };
    float Gy[9] = { -1, -2, -1,
                     0,  0,  0,
                     1,  2,  1 };
 
    // Greyscale conversion using perceptive coefficients to get a single intensity value per pixel for edge detection
    // Result is stored in the R plane of grayImg, G and B planes are not used
    Image* grayImg = allocImage(width, height);
    for (int i = 0; i < planeSize; i++) {
        grayImg->data[i] = static_cast<uint8_t>(
            0.299f * input->data[i] +  // R
            0.587f * input->data[i + planeSize] +  // G
            0.114f * input->data[i + 2 * planeSize]    // B
        );
    }
 
    Image* paddedImg = addPadding(grayImg, 1);
    int paddedWidth = paddedImg->width;
 
    // gx and gy are the horizontal and vertical gradients computed by convolving the 3x3 region with Gx and Gy respectively
    // The magnitude of the gradient vector (gx, gy) gives the edge strength at that pixel
    // The output is a single intensity value representing edge strength, stored in all three channels of the output image for visualization
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float gx = 0.f, gy = 0.f;
            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    float px = paddedImg->data[(y + ky) * paddedWidth + (x + kx)];
                    gx += px * Gx[ky * 3 + kx];
                    gy += px * Gy[ky * 3 + kx];
                }
            }

            // Magnitude is the length of the gradient vector, clamped to 255 for uint8_t representation
            uint8_t magnitude = static_cast<uint8_t>(std::min(std::sqrt(gx * gx + gy * gy), 255.f));
            int idx = y * width + x;
            output->data[idx] = magnitude; // R
            output->data[idx + planeSize] = magnitude; // G
            output->data[idx + 2 * planeSize] = magnitude; // B
        }
    }
 
    freeImage(grayImg);
    freeImage(paddedImg);
}