#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define FILTER_RADIUS 2
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

// Constant memory for filter (up to 16KB)
#define MAX_FILTER_SIZE 25
__constant__ float F_const[MAX_FILTER_SIZE];

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// ============================================================================
// Kernel 1: Basic global memory convolution
// ============================================================================
__global__ void convolution_2D_global_memory_kernel(
    float *N, float *P, int width, int height, int filter_radius)
{
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    
    float Pvalue = 0.0f;
    
    for (int fRow = 0; fRow < 2*filter_radius+1; fRow++) {
        for (int fCol = 0; fCol < 2*filter_radius+1; fCol++) {
            int inRow = outRow - filter_radius + fRow;
            int inCol = outCol - filter_radius + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F_const[fRow * (2*filter_radius+1) + fCol] * 
                          N[inRow * width + inCol];
            }
        }
    }
    
    if (outRow < height && outCol < width) {
        P[outRow * width + outCol] = Pvalue;
    }
}

// ============================================================================
// Kernel 2: Constant memory with basic tiling
// ============================================================================
__global__ void convolution_2D_const_mem_kernel(
    float *N, float *P, int width, int height)
{
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y;
    
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    
    if (row < height && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    __syncthreads();
    
    // Calculate output only for the interior threads
    if (threadIdx.y >= FILTER_RADIUS && 
        threadIdx.y < IN_TILE_DIM - FILTER_RADIUS &&
        threadIdx.x >= FILTER_RADIUS && 
        threadIdx.x < IN_TILE_DIM - FILTER_RADIUS) {
        
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
            for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                Pvalue += F_const[fRow * (2*FILTER_RADIUS+1) + fCol] *
                          N_s[threadIdx.y - FILTER_RADIUS + fRow]
                             [threadIdx.x - FILTER_RADIUS + fCol];
            }
        }
        
        int outRow = row;
        int outCol = col;
        if (outRow < height && outCol < width) {
            P[outRow * width + outCol] = Pvalue;
        }
    }
}

// ============================================================================
// Kernel 3: Cached tiling with halo and constant memory
// ============================================================================
__global__ void convolution_cached_tiled_2D_const_mem_kernel(
    float *N, float *P, int width, int height)
{
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y;
    
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    
    // Load input tile with halo
    if (row >= 0 && row < height && col >= 0 && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    __syncthreads();
    
    // Calculate output only for the interior threads
    if (col < width && row < height) {
        if (threadIdx.y >= FILTER_RADIUS && 
            threadIdx.y < IN_TILE_DIM - FILTER_RADIUS &&
            threadIdx.x >= FILTER_RADIUS && 
            threadIdx.x < IN_TILE_DIM - FILTER_RADIUS) {
            
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
                for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                    if ((threadIdx.x - FILTER_RADIUS + fCol >= 0) &&
                        (threadIdx.x - FILTER_RADIUS + fCol < IN_TILE_DIM) &&
                        (threadIdx.y - FILTER_RADIUS + fRow >= 0) &&
                        (threadIdx.y - FILTER_RADIUS + fRow < IN_TILE_DIM)) {
                        Pvalue += F_const[fRow * (2*FILTER_RADIUS+1) + fCol] *
                                  N_s[threadIdx.y - FILTER_RADIUS + fRow]
                                     [threadIdx.x - FILTER_RADIUS + fCol];
                    }
                }
            }
            
            if (row < height && col < width) {
                P[row * width + col] = Pvalue;
            }
        }
    }
}

// ============================================================================
// Filter Definitions
// ============================================================================
enum FilterType {
    FILTER_IDENTITY,
    FILTER_GAUSSIAN_BLUR,
    FILTER_SOBEL_X,
    FILTER_SOBEL_Y,
    FILTER_SHARPEN,
    FILTER_EDGE_DETECT
};

void createFilter(FilterType type, float* h_F, float& scale) {
    memset(h_F, 0, MAX_FILTER_SIZE * sizeof(float));
    
    switch (type) {
        case FILTER_IDENTITY:
            // Identity: pass through
            h_F[2*5 + 2] = 1.0f;
            scale = 1.0f;
            break;
            
        case FILTER_GAUSSIAN_BLUR:
            // Gaussian blur 5x5
            {
                float gaussian[] = {
                    1, 4, 6, 4, 1,
                    4, 16, 24, 16, 4,
                    6, 24, 36, 24, 6,
                    4, 16, 24, 16, 4,
                    1, 4, 6, 4, 1
                };
                memcpy(h_F, gaussian, 25 * sizeof(float));
                scale = 256.0f;
            }
            break;
            
        case FILTER_SOBEL_X:
            // Sobel X (detect vertical edges)
            {
                float sobel_x[] = {
                    -1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0
                };
                memcpy(h_F, sobel_x, 25 * sizeof(float));
                scale = 1.0f;
            }
            break;
            
        case FILTER_SOBEL_Y:
            // Sobel Y (detect horizontal edges)
            {
                float sobel_y[] = {
                    -1, -2, -1,
                    0, 0, 0,
                    1, 2, 1,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0
                };
                memcpy(h_F, sobel_y, 25 * sizeof(float));
                scale = 1.0f;
            }
            break;
            
        case FILTER_SHARPEN:
            // Sharpen
            {
                float sharpen[] = {
                    0, -1, 0,
                    -1, 5, -1,
                    0, -1, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0
                };
                memcpy(h_F, sharpen, 25 * sizeof(float));
                scale = 1.0f;
            }
            break;
            
        case FILTER_EDGE_DETECT:
            // General edge detection
            {
                float edge[] = {
                    -1, -1, -1,
                    -1, 8, -1,
                    -1, -1, -1,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0
                };
                memcpy(h_F, edge, 25 * sizeof(float));
                scale = 1.0f;
            }
            break;
    }
}

const char* getFilterName(FilterType type) {
    switch (type) {
        case FILTER_IDENTITY: return "Identity";
        case FILTER_GAUSSIAN_BLUR: return "Gaussian Blur";
        case FILTER_SOBEL_X: return "Sobel X";
        case FILTER_SOBEL_Y: return "Sobel Y";
        case FILTER_SHARPEN: return "Sharpen";
        case FILTER_EDGE_DETECT: return "Edge Detection";
        default: return "Unknown";
    }
}

// ============================================================================
// Convolution launchers
// ============================================================================
void launchConvolution_GlobalMemory(
    float* h_N, float* h_P, int width, int height,
    float* h_F, float scale)
{
    float *d_N, *d_P;
    size_t bytes = width * height * sizeof(float);
    
    cudaMalloc(&d_N, bytes);
    cudaMalloc(&d_P, bytes);
    
    cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F_const, h_F, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float));
    
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    convolution_2D_global_memory_kernel<<<grid, block>>>(d_N, d_P, width, height, FILTER_RADIUS);
    checkCuda(cudaGetLastError(), "Global memory kernel");
    checkCuda(cudaDeviceSynchronize(), "Global memory sync");
    
    cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);
    
    // Normalize
    for (int i = 0; i < width * height; i++) {
        h_P[i] /= scale;
    }
    
    cudaFree(d_N);
    cudaFree(d_P);
}

void launchConvolution_ConstMemory(
    float* h_N, float* h_P, int width, int height,
    float* h_F, float scale)
{
    float *d_N, *d_P;
    size_t bytes = width * height * sizeof(float);
    
    cudaMalloc(&d_N, bytes);
    cudaMalloc(&d_P, bytes);
    
    cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F_const, h_F, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float));
    
    dim3 block(IN_TILE_DIM, IN_TILE_DIM);
    dim3 grid((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, 
              (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    
    convolution_2D_const_mem_kernel<<<grid, block>>>(d_N, d_P, width, height);
    checkCuda(cudaGetLastError(), "Const memory kernel");
    checkCuda(cudaDeviceSynchronize(), "Const memory sync");
    
    cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);
    
    // Normalize
    for (int i = 0; i < width * height; i++) {
        h_P[i] /= scale;
    }
    
    cudaFree(d_N);
    cudaFree(d_P);
}

void launchConvolution_CachedTiled(
    float* h_N, float* h_P, int width, int height,
    float* h_F, float scale)
{
    float *d_N, *d_P;
    size_t bytes = width * height * sizeof(float);
    
    cudaMalloc(&d_N, bytes);
    cudaMalloc(&d_P, bytes);
    
    cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F_const, h_F, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float));
    
    dim3 block(IN_TILE_DIM, IN_TILE_DIM);
    dim3 grid((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, 
              (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    
    convolution_cached_tiled_2D_const_mem_kernel<<<grid, block>>>(d_N, d_P, width, height);
    checkCuda(cudaGetLastError(), "Cached tiled kernel");
    checkCuda(cudaDeviceSynchronize(), "Cached tiled sync");
    
    cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);
    
    // Normalize
    for (int i = 0; i < width * height; i++) {
        h_P[i] /= scale;
    }
    
    cudaFree(d_N);
    cudaFree(d_P);
}

// ============================================================================
// Verification
// ============================================================================
bool validateResults(float* P1, float* P2, int width, int height, const char* name1, const char* name2) {
    float maxDiff = 0.0f;
    for (int i = 0; i < width * height; i++) {
        float diff = fabsf(P1[i] - P2[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    
    printf("  %s vs %s: maxDiff = %e\n", name1, name2, maxDiff);
    return maxDiff < 1e-3f;
}

// ============================================================================
// Image I/O
// ============================================================================
float* loadImage(const char* filename, int* width, int* height, int* channels) {
    unsigned char* data = stbi_load(filename, width, height, channels, 1);
    if (!data) {
        fprintf(stderr, "Error: Cannot load image %s\n", filename);
        return NULL;
    }
    
    // Convert to grayscale float
    int size = (*width) * (*height);
    float* fdata = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        fdata[i] = (float)data[i] / 255.0f;
    }
    stbi_image_free(data);
    return fdata;
}

bool saveImage(const char* filename, float* data, int width, int height) {
    // Convert float to unsigned char (single channel grayscale)
    unsigned char* bdata = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    for (int i = 0; i < width * height; i++) {
        float val = data[i];
        // Clamp to [0, 1]
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        bdata[i] = (unsigned char)(val * 255.0f);
    }
    
    int result = stbi_write_png(filename, width, height, 1, bdata, width);
    free(bdata);
    
    if (!result) {
        fprintf(stderr, "Error: Cannot save image to %s\n", filename);
        return false;
    }
    printf("Saved: %s\n", filename);
    return true;
}

// ============================================================================
// Command Line Arguments Parser
// ============================================================================
void printUsage(const char* program_name) {
    printf("\n");
    printf("====================================================================\n");
    printf("Usage: %s [mode] [args...]\n", program_name);
    printf("====================================================================\n");
    printf("\nMode 1 - Synthetic Image (Legacy):\n");
    printf("  %s [filter_id] [kernel_id]\n", program_name);
    printf("  filter_id: 0=Identity, 1=Gaussian, 2=Sobel-X, 3=Sobel-Y, 4=Sharpen, 5=Edge\n");
    printf("  kernel_id: 0=Global, 1=Const, 2=Cached (default)\n");
    printf("  Example: %s 1 2\n", program_name);
    printf("           (Gaussian blur with cached tiled kernel)\n");
    printf("\nMode 2 - Load and Process Image:\n");
    printf("  %s input.png output.png [filter_id] [kernel_id]\n", program_name);
    printf("  input.png:  Path to input image (PNG/JPG/etc)\n");
    printf("  output.png: Path to output image (PNG format)\n");
    printf("  filter_id:  0=Identity, 1=Gaussian, 2=Sobel-X, 3=Sobel-Y, 4=Sharpen, 5=Edge\n");
    printf("  kernel_id:  0=Global, 1=Const, 2=Cached (default)\n");
    printf("  Example: %s photo.jpg result.png 1 2\n", program_name);
    printf("           (Apply Gaussian blur to photo.jpg and save to result.png)\n");
    printf("====================================================================\n\n");
}

struct ConvolutionParams {
    FilterType filter;
    int kernel_type;
    const char* input_file;
    const char* output_file;
    bool from_file;
    int width;
    int height;
};

ConvolutionParams parseArguments(int argc, char** argv) {
    ConvolutionParams params;
    params.filter = FILTER_GAUSSIAN_BLUR;
    params.kernel_type = 2;  // 0=global, 1=const, 2=cached (default)
    params.input_file = NULL;
    params.output_file = NULL;
    params.from_file = false;
    params.width = 512;
    params.height = 512;
    
    if (argc < 2) {
        return params;  // Use defaults
    }
    
    // Check if first arg is numeric (legacy mode) or filename
    if (argv[1][0] >= '0' && argv[1][0] <= '9' && strlen(argv[1]) == 1) {
        // Legacy mode: convolution.exe filter_id [kernel_id]
        int f = atoi(argv[1]);
        if (f >= 0 && f < 6) {
            params.filter = (FilterType)f;
        }
        if (argc >= 3) {
            params.kernel_type = atoi(argv[2]);
        }
        params.from_file = false;
    } else {
        // New mode: convolution.exe input.png output.png [filter_id] [kernel_id]
        params.input_file = argv[1];
        if (argc >= 3) {
            params.output_file = argv[2];
        }
        if (argc >= 4) {
            int f = atoi(argv[3]);
            if (f >= 0 && f < 6) {
                params.filter = (FilterType)f;
            }
        }
        if (argc >= 5) {
            params.kernel_type = atoi(argv[4]);
        }
        params.from_file = true;
    }
    
    return params;
}
// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    // Print usage info
    printUsage(argv[0]);
    
    // Parse command line arguments
    ConvolutionParams params = parseArguments(argc, argv);
    
    if (!params.from_file) {
        params.width = 512;
        params.height = 512;
    }
    
    printf("2D Convolution Test\n");
    printf("Kernel: %d (0=Global, 1=Const, 2=Cached)\n", params.kernel_type);
    printf("Filter: %s\n", getFilterName(params.filter));
    printf("\n");
    
    float *h_N, *h_P;
    
    if (params.from_file && params.input_file) {
        // Load from file
        int channels;
        h_N = loadImage(params.input_file, &params.width, &params.height, &channels);
        if (!h_N) {
            return 1;
        }
        printf("Loaded: %s (%dx%d)\n", params.input_file, params.width, params.height);
    } else {
        // Generate synthetic pattern
        printf("Image: %dx%d (synthetic)\n", params.width, params.height);
        size_t bytes = params.width * params.height * sizeof(float);
        h_N = (float*)malloc(bytes);
        for (int i = 0; i < params.width * params.height; i++) {
            h_N[i] = sinf(i / 100.0f) + cosf(i / 50.0f);
        }
    }
    
    size_t bytes = params.width * params.height * sizeof(float);
    h_P = (float*)malloc(bytes);
    float *h_F = (float*)malloc(MAX_FILTER_SIZE * sizeof(float));
    
    float scale;
    createFilter(params.filter, h_F, scale);
    
    printf("Running convolution...\n");
    
    // Run selected kernel
    if (params.kernel_type == 0) {
        launchConvolution_GlobalMemory(h_N, h_P, params.width, params.height, h_F, scale);
        printf("  Global memory kernel: OK\n");
    } else if (params.kernel_type == 1) {
        launchConvolution_ConstMemory(h_N, h_P, params.width, params.height, h_F, scale);
        printf("  Constant memory kernel: OK\n");
    } else {
        launchConvolution_CachedTiled(h_N, h_P, params.width, params.height, h_F, scale);
        printf("  Cached tiled kernel: OK\n");
    }
    
    // Save if output file specified
    if (params.from_file && params.output_file) {
        saveImage(params.output_file, h_P, params.width, params.height);
    }
    
    printf("\nCompleted!\n");
    
    free(h_N);
    free(h_P);
    free(h_F);
    
    return 0;
}
