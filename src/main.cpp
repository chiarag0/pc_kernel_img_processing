#include <iostream>
#include <chrono>
#include "ImageIO.h"
#include "Sequential.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Error: no input image specified\n";
        return 1;
    }

    Image* input = loadImage(argv[1]);
    if (!input) return 1;
    std::cout << "Image loaded: " << input->width << "x" << input->height << "\n";

    Image* output = allocImage(input->width, input->height);

    // Test Gaussian 5x5 (size=5, sigma=1.0)
    Kernel gauss = makeKernel(KernelType::GAUSSIAN, 5, 1.0f);
    auto t0 = std::chrono::high_resolution_clock::now();
    sequentialConvolution(input, output, gauss);
    auto t1 = std::chrono::high_resolution_clock::now();
    double msGauss = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Sequential Gaussian 5x5: " << msGauss << " ms\n";
    saveImage(output, "output_gaussian.png");

    // Test Sobel
    auto t2 = std::chrono::high_resolution_clock::now();
    sequentialSobel(input, output);
    auto t3 = std::chrono::high_resolution_clock::now();
    double msSobel = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << "Sequential Sobel: " << msSobel << " ms\n";
    saveImage(output, "output_sobel.png");

    freeKernel(gauss);
    freeImage(input);
    freeImage(output);
    return 0;
}