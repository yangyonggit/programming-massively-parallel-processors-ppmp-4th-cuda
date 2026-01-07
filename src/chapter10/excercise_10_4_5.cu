#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>

#define ARRAY_SIZE 4096
#define COARSE_FACTOR 4
#define BLOCK_DIM (ARRAY_SIZE / (COARSE_FACTOR * 2))
#include <cmath>   // for INFINITY

// Max Reduction Kernel
__global__ void ConvergentMaxReductionKernel(float* input, float* output) {
    unsigned int i = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] = fmaxf(input[i], input[i + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

__device__ float atomicMaxFloat(float* addr, float value) {
    int* addr_i = (int*)addr;
    int old = *addr_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_i, assumed,
            __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// Coarsened Sum Reduction Kernel (processes multiple tiles per thread block)
__global__ void CoarsenedMaxReductionKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    float maxVal = input[i];
    for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile) {
        maxVal = fmaxf(maxVal, input[i + tile * BLOCK_DIM]);
    }
    input_s[t] = maxVal;
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] = fmaxf(input_s[t], input_s[t + stride]);
        }
    }
    if (t == 0) {
        atomicMaxFloat(output, input_s[0]);
    }
}


// Coarsened Sum Reduction Kernel (processes multiple tiles per thread block)
__global__ void CoarsenedMaxReductionKernelaAbitraryLength(float* input, float* output, int N) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    unsigned int base = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i0 = base + t;
 

    float maxVal = -INFINITY;
    #pragma unroll
    for (unsigned int tile = 0; tile < COARSE_FACTOR * 2; ++tile) {
        unsigned int idx = i0 + tile * blockDim.x;   // 用 blockDim.x
        float v = (idx < (unsigned)N) ? input[idx] : -INFINITY;
        maxVal = fmaxf(maxVal, v);
    }
    
    input_s[t] = maxVal;
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        __syncthreads();
        if (t < stride) {
            input_s[t] = fmaxf(input_s[t], input_s[t + stride]);
        }
    }
    if (t == 0) {
        atomicMaxFloat(output, input_s[0]);
    }
}

// CPU reference implementation for max
float cpuMaxReduction(float* input, int size) {
    float maxVal = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > maxVal) {
            maxVal = input[i];
        }
    }
    return maxVal;
}

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Run arbitrary-length test using the arbitrary-length kernel
void runArbitraryLengthTest(const char* title, int N) {
    int threadsPerBlock = BLOCK_DIM;
    int blocksPerGrid = (N + (COARSE_FACTOR * 2 * threadsPerBlock) - 1) /
        (COARSE_FACTOR * 2 * threadsPerBlock);
    int bytes = N * sizeof(float);

    float* h_input = (float*)malloc(bytes);
    float h_output = -INFINITY;
    float cpu_result;

    // Craft data with a clear maximum at an irregular position
    for (int i = 0; i < N; ++i) {
        h_input[i] = -500.0f + (float)((i * 37) % 1000) / 10.0f;
    }
    if (N > 3) {
        h_input[(N * 3) / 5] = 12345.0f;  // ensure a distinct max
    }

    cpu_result = cpuMaxReduction(h_input, N);

    float* d_input;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, -INFINITY, sizeof(float)));

    CoarsenedMaxReductionKernelaAbitraryLength<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    bool correct = fabs(cpu_result - h_output) < 1e-5f;
    printf("Arbitrary Test (N=%d): %s\n", N, title);
    printf("CPU Result: %.2f\n", cpu_result);
    printf("GPU Result: %.2f\n", h_output);
    printf("Status: %s\n\n", correct ? "PASS" : "FAIL");

    free(h_input);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    // Array size (must be power of 2 for this kernel)
    int size = ARRAY_SIZE;
    int bytes = size * sizeof(float);
    
    // Allocate host memory
    float* h_input = (float*)malloc(bytes);
    float h_output = -INFINITY;
    
    // Allocate device memory
    float* d_input;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    
    int threadsPerBlock = BLOCK_DIM;
    int blocksPerGrid = (size + (COARSE_FACTOR * 2 * threadsPerBlock) - 1) /
        (COARSE_FACTOR * 2 * threadsPerBlock);
    float epsilon = 1e-5f;
    bool correct;
    float cpu_result;
    
    printf("========== MAX REDUCTION KERNEL TESTS ==========\n\n");
    
    // Test 1: All same values
    printf("Test 1: All elements = 5.0\n");
    for (int i = 0; i < size; i++) {
        h_input[i] = 5.0f;
    }
    
    cpu_result = cpuMaxReduction(h_input, size);
    printf("CPU Result: %.2f\n", cpu_result);
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, -INFINITY, sizeof(float)));
    CoarsenedMaxReductionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("GPU Result: %.2f\n", h_output);
    correct = fabs(cpu_result - h_output) < epsilon;
    printf("Status: %s\n\n", correct ? "PASS" : "FAIL");
    
    // Test 2: Sequential values (max should be at end)
    printf("Test 2: Sequential values (0, 1, 2, ..., %d)\n", size - 1);
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)i;
    }
    
    cpu_result = cpuMaxReduction(h_input, size);
    printf("CPU Result: %.2f\n", cpu_result);
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, -INFINITY, sizeof(float)));
    CoarsenedMaxReductionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("GPU Result: %.2f\n", h_output);
    correct = fabs(cpu_result - h_output) < epsilon;
    printf("Status: %s\n\n", correct ? "PASS" : "FAIL");
    
    // Test 3: Reverse sequential (max should be at beginning)
    printf("Test 3: Reverse sequential (%d, %d, ..., 1, 0)\n", size - 1, size - 2);
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)(size - 1 - i);
    }
    
    cpu_result = cpuMaxReduction(h_input, size);
    printf("CPU Result: %.2f\n", cpu_result);
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, -INFINITY, sizeof(float)));
    CoarsenedMaxReductionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("GPU Result: %.2f\n", h_output);
    correct = fabs(cpu_result - h_output) < epsilon;
    printf("Status: %s\n\n", correct ? "PASS" : "FAIL");
    
    // Test 4: Random values
    printf("Test 4: Random values\n");
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)(rand() % 10000) / 10.0f;
    }
    
    cpu_result = cpuMaxReduction(h_input, size);
    printf("CPU Result: %.2f\n", cpu_result);
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, -INFINITY, sizeof(float)));
    CoarsenedMaxReductionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("GPU Result: %.2f\n", h_output);
    correct = fabs(cpu_result - h_output) < epsilon;
    printf("Status: %s\n\n", correct ? "PASS" : "FAIL");
    
    // Test 5: Max in middle
    printf("Test 5: Max value in the middle\n");
    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f;
    }
    h_input[size / 2] = 999.0f;  // Put max in the middle
    
    cpu_result = cpuMaxReduction(h_input, size);
    printf("CPU Result: %.2f\n", cpu_result);
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, -INFINITY, sizeof(float)));
    CoarsenedMaxReductionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("GPU Result: %.2f\n", h_output);
    correct = fabs(cpu_result - h_output) < epsilon;
    printf("Status: %s\n\n", correct ? "PASS" : "FAIL");
    
    // Test 6: Negative values
    printf("Test 6: Negative values\n");
    for (int i = 0; i < size; i++) {
        h_input[i] = -100.0f + (float)i / 10.0f;
    }
    
    cpu_result = cpuMaxReduction(h_input, size);
    printf("CPU Result: %.2f\n", cpu_result);
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, -INFINITY, sizeof(float)));
    CoarsenedMaxReductionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("GPU Result: %.2f\n", h_output);
    correct = fabs(cpu_result - h_output) < epsilon;
    printf("Status: %s\n\n", correct ? "PASS" : "FAIL");
    
    printf("================================================\n");

    // Arbitrary-length tests (sizes not tied to ARRAY_SIZE)
    runArbitraryLengthTest("Non power-of-two length", 5000);
    runArbitraryLengthTest("Another irregular length", 7777);
    
    // Cleanup
    free(h_input);
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\nAll tests completed!\n");
    return 0;
}
