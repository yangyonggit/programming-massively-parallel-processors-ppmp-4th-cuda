#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "cdiv.h"

// Radius of the blur window; total window size is (2 * BLUR_SIZE + 1)^2
#define BLUR_SIZE 9

__global__ void blurKernel(const unsigned char* in, unsigned char* out, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < w && row < h) {
        const int CHANNELS = 3;
        int pixVal[CHANNELS] = {0, 0, 0};
        int pixels = 0;

        // Average over surrounding BLUR_SIZE x BLUR_SIZE box
        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                // Verify we have a valid pixel
                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    int rgbOffset = (curRow * w + curCol) * CHANNELS;
                    pixVal[0] += in[rgbOffset];
                    pixVal[1] += in[rgbOffset + 1];
                    pixVal[2] += in[rgbOffset + 2];
                    ++pixels; // Track number of pixels in the average
                }
            }
        }

        // Write blurred pixel
        int outOffset = (row * w + col) * CHANNELS;
        out[outOffset]     = static_cast<unsigned char>(pixVal[0] / pixels);
        out[outOffset + 1] = static_cast<unsigned char>(pixVal[1] / pixels);
        out[outOffset + 2] = static_cast<unsigned char>(pixVal[2] / pixels);
    }
}


void launchBlur(const unsigned char* h_input, unsigned char* h_output, int width, int height) {
    const size_t numPixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t bytes = numPixels * 3;

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    blurKernel<<<dimGrid, dimBlock>>>(d_input, d_output, width, height);

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image>" << std::endl;
        return 1;
    }

    const char* input_filename = argv[1];
    int width = 0, height = 0, channels = 0;

    // Load as 3-channel RGB
    unsigned char* h_input = stbi_load(input_filename, &width, &height, &channels, 3);
    if (h_input == nullptr) {
        std::cerr << "Failed to load image: " << input_filename << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << width << "x" << height << " (" << channels << " channels -> 3)" << std::endl;

    unsigned char* h_output = static_cast<unsigned char*>(malloc(width * height * 3));
    if (h_output == nullptr) {
        std::cerr << "Failed to allocate output buffer" << std::endl;
        stbi_image_free(h_input);
        return 1;
    }

    launchBlur(h_input, h_output, width, height);

    const char* output_filename = "output_blur.png";
    int success = stbi_write_png(output_filename, width, height, 3, h_output, width * 3);
    if (success) {
        std::cout << "Output saved to: " << output_filename << std::endl;
    } else {
        std::cerr << "Failed to save output image" << std::endl;
    }

    stbi_image_free(h_input);
    free(h_output);

    return success ? 0 : 1;
}
