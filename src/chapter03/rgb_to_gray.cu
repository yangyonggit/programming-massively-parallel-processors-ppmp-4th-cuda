#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

__global__ void rgbToGrayscaleKernel(const unsigned char* Pin, unsigned char* Pout, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    const int CHANNELS = 3;

    if (col < width && row < height) {
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * CHANNELS;

        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        Pout[grayOffset] = static_cast<unsigned char>(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

void launchRgbToGray(const unsigned char* h_input, unsigned char* h_output, int width, int height) {
    const size_t numPixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t inBytes = numPixels * 3;
    const size_t outBytes = numPixels;

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    cudaMalloc(&d_input, inBytes);
    cudaMalloc(&d_output, outBytes);

    cudaMemcpy(d_input, h_input, inBytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(32, 32);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    rgbToGrayscaleKernel<<<dimGrid, dimBlock>>>(d_input, d_output, width, height);

    cudaMemcpy(h_output, d_output, outBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image>" << std::endl;
        return 1;
    }

    const char* input_filename = argv[1];
    int width, height, channels;

    // Load image
    unsigned char* h_input = stbi_load(input_filename, &width, &height, &channels, 3);
    if (h_input == nullptr) {
        std::cerr << "Failed to load image: " << input_filename << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << width << "x" << height << " (" << channels << " channels)" << std::endl;

    // Allocate output buffer for grayscale (1 channel)
    unsigned char* h_output = static_cast<unsigned char*>(malloc(width * height * sizeof(unsigned char)));
    if (h_output == nullptr) {
        std::cerr << "Failed to allocate output buffer" << std::endl;
        stbi_image_free(h_input);
        return 1;
    }

    // Call CUDA kernel launcher
    launchRgbToGray(h_input, h_output, width, height);

    // Save output image
    const char* output_filename = "output_gray.png";
    int success = stbi_write_png(output_filename, width, height, 1, h_output, width);
    if (success) {
        std::cout << "Output saved to: " << output_filename << std::endl;
    } else {
        std::cerr << "Failed to save output image" << std::endl;
    }

    // Cleanup
    stbi_image_free(h_input);
    free(h_output);

    return success ? 0 : 1;
}



