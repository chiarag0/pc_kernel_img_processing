#pragma once
#include <string>
#include <cstdint>

struct Image {
    int width;
    int height;
    int channels; // RGB
    uint8_t* data;  // planar array: R plane - G plane - B plane
};

// Returns the size of one color plane (R, G, or B) in bytes
int planeSize(const Image* img);

// Accessors for the R, G, B planes of the image data
uint8_t* redChannel(const Image* img);
uint8_t* greenChannel(const Image* img);
uint8_t* blueChannel(const Image* img);

// Load image from file path and convert it to planar format (SoA)
Image* loadImage(const std::string& path);

// Convert image to AoS and save it to file path
bool saveImage(const Image* img, const std::string& path);

// Allocate memory for an image of given width and height, with 3 channels (RGB).
Image* allocImage(int width, int height);

void freeImage(Image* img);

// Add padding to the image by replicating the border pixels
// New size will be (width + 2*padding) x (height + 2*padding)
Image* addPadding(const Image* img, int padding);.