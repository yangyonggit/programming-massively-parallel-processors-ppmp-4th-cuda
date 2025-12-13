#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int deviceCount = 0;

    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    printf("Number of CUDA devices: %d\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("========================================\n");
        printf("Device %d: %s\n", dev, prop.name);

        printf("  Compute Capability        : %d.%d\n",
               prop.major, prop.minor);

        printf("  SM Count                  : %d\n",
               prop.multiProcessorCount);

        printf("  Warp Size                 : %d\n",
               prop.warpSize);

        printf("  Max Threads per Block     : %d\n",
               prop.maxThreadsPerBlock);

        printf("  Max Threads per SM        : %d\n",
               prop.maxThreadsPerMultiProcessor);

        printf("  Max Blocks per SM         : %d\n",
               prop.maxBlocksPerMultiProcessor);

        printf("  Registers per SM          : %d\n",
               prop.regsPerMultiprocessor);

        printf("  Shared Mem per SM (KB)    : %zu\n",
               prop.sharedMemPerMultiprocessor / 1024);

        printf("  Shared Mem per Block (KB) : %zu\n",
               prop.sharedMemPerBlock / 1024);

        printf("  Max Grid Size             : (%d, %d, %d)\n",
               prop.maxGridSize[0],
               prop.maxGridSize[1],
               prop.maxGridSize[2]);

        printf("  Max Threads Dim           : (%d, %d, %d)\n",
               prop.maxThreadsDim[0],
               prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);

        printf("  Clock Rate (MHz)          : %.1f\n",
               prop.clockRate / 1000.0);

        printf("========================================\n\n");
    }

    return 0;
}
