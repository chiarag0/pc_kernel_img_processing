#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"
#include "ImageIO.h"
#include <cstring>
#include <iostream>

int planeSize(const Image* img) {
    return img->width * img->height;
}

uint8_t* redChannel(const Image* img) { return img->data; }
uint8_t* greenChannel(const Image* img) { return img->data + planeSize(img); }
uint8_t* blueChannel(const Image* img) { return img->data + 2 * planeSize(img); }

Image* loadImage(const std::string& path) {
    int width, height, channels;

    // stb_image loads images in AoS format: [R0,G0,B0, R1,G1,B1, ...]
    uint8_t* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (!data) {
        std::cerr << "Failed to load image: " << path << "\n";
        return nullptr;
    }

    Image* img = allocImage(width, height);

    // AoS → SoA conversion: [R0,R1,..., G0,G1,..., B0,B1,...]
    int n = width * height;
    int pSize = planeSize(img);
    for (int i = 0; i < n; i++) {
        img->data[i] = data[i * 3 + 0];
        img->data[i + pSize] = data[i * 3 + 1];
        img->data[i + 2 * pSize] = data[i * 3 + 2];
    }

    // Free the original data loaded by stb_image
    stbi_image_free(data);
    return img;
}

bool saveImage(const Image* img, const std::string& path) {
    // SoA → AoS conversion for stb_image_write
    int pSize = planeSize(img);    
    uint8_t* data = new uint8_t[pSize * 3];
    for (int i = 0; i < pSize; i++) {
        data[i * 3 + 0] = img->data[i];
        data[i * 3 + 1] = img->data[i + pSize];
        data[i * 3 + 2] = img->data[i + 2 * pSize];
    }

    int result = stbi_write_png(path.c_str(), img->width, img->height, 3, data, img->width * 3);
    delete[] data;
    return result != 0;
}

Image* allocImage(int width, int height) {
    Image* img = new Image();
    img->width    = width;
    img->height   = height;
    img->channels = 3;
    
    // Init to zero 
    img->data = new uint8_t[width * height * 3]();
    return img;
}

void freeImage(Image* img) {
    if (!img) return;
    delete[] img->data;
    delete img;
}

// Zero padding by replicating border pixels initiated to zero in allocImage
Image* addPadding(const Image* img, int padding) {
    int paddedWidth = img->width  + 2 * padding;
    int paddedHeight = img->height + 2 * padding;

    Image* paddedImg = allocImage(paddedWidth, paddedHeight);
    int srcPlaneSize = planeSize(img);
    int dstPlaneSize = planeSize(paddedImg);

    // Copy original image into the center of the padded image, channel by channel
    // Row i of the original goes to row (i + padding) of the padded, starting at column 'padding'
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < img->height; i++) {
            const uint8_t* src = img->data + c * srcPlaneSize + i * img->width;
            uint8_t* dst = paddedImg->data + c * dstPlaneSize + (i + padding) * paddedWidth + padding;
            std::memcpy(dst, src, img->width);
        }
    }

    return paddedImg;
}