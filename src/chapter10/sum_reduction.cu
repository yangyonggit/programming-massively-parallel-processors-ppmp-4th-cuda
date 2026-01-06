#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ARRAY_SIZE 2048
#define BLOCK_DIM (ARRAY_SIZE >> 1)

// Simple Sum Reduction Kernel
__global__ void SimpleSumReductionKernel(float* input, float* output) {
    unsigned int i = 2 * threadIdx.x;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

__global__ void ConvergentSumReductionKernel(float* input, float* output) {
    unsigned int i = 2 * threadIdx.x;
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

__global__ void SharedMemorySumReductionKernel(float* input, float* output) {
    __shared__ float sdata[BLOCK_DIM];
    unsigned int tid = threadIdx.x;
    unsigned int i = tid + BLOCK_DIM;
    
    sdata[tid] = input[i] + input[i + 1];
  
    
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
         __syncthreads();
        if (tid < stride) {
            
            sdata[tid] += sdata[tid + stride];
        }       
    }
    
    if (tid == 0) {
        *output = sdata[0];
    }
}

// CPU reference implementation for verification
float cpuSumReduction(float* input, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += input[i];
    }
    return sum;
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

int main() {
    // Array size (must be power of 2 for this simple kernel)
    int size = 2048;
    int bytes = size * sizeof(float);
    
    // Allocate host memory
    float* h_input = (float*)malloc(bytes);
    float h_output;
    
    // Initialize input array
    printf("Initializing array with %d elements...\n", size);
    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f;  // Simple test: all 1s, sum should be size
    }
    
    // Calculate CPU reference result
    printf("Computing CPU reference result...\n");
    float cpu_result = cpuSumReduction(h_input, size);
    printf("CPU Result: %.2f\n", cpu_result);
    
    // Allocate device memory
    float* d_input;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    // Launch kernel
    // Number of threads should be size/2 since each thread processes 2 elements
    int threadsPerBlock = size / 2;
    printf("\nLaunching kernel with %d threads...\n", threadsPerBlock);
    
    SimpleSumReductionKernel<<<1, threadsPerBlock>>>(d_input, d_output);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("GPU Result: %.2f\n", h_output);
    
    // Verify results
    float epsilon = 1e-3f;
    bool correct = fabs(cpu_result - h_output) < epsilon;
    
    printf("\n========== VERIFICATION ==========\n");
    printf("Expected (CPU): %.2f\n", cpu_result);
    printf("Got (GPU):      %.2f\n", h_output);
    printf("Difference:     %.6f\n", fabs(cpu_result - h_output));
    printf("Status:         %s\n", correct ? "PASS" : "FAIL");
    printf("==================================\n");
    
    // Test with different values
    printf("\n\nTesting with different input values...\n");
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)i;  // 0, 1, 2, 3, ...
    }
    
    cpu_result = cpuSumReduction(h_input, size);
    printf("CPU Result: %.2f\n", cpu_result);
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    SimpleSumReductionKernel<<<1, threadsPerBlock>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("GPU Result: %.2f\n", h_output);
    correct = fabs(cpu_result - h_output) < epsilon;
    printf("Status: %s (difference: %.6f)\n", correct ? "PASS" : "FAIL", 
           fabs(cpu_result - h_output));
    
    // Test ConvergentSumReductionKernel
    printf("\n\n========== TESTING CONVERGENT KERNEL ==========\n");
    
    // Test 1: All 1s
    printf("\nTest 1: All elements = 1.0\n");
    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f;
    }
    
    cpu_result = cpuSumReduction(h_input, size);
    printf("CPU Result: %.2f\n", cpu_result);
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    ConvergentSumReductionKernel<<<1, threadsPerBlock>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("GPU Result (Convergent): %.2f\n", h_output);
    correct = fabs(cpu_result - h_output) < epsilon;
    printf("Status: %s (difference: %.6f)\n", correct ? "PASS" : "FAIL", 
           fabs(cpu_result - h_output));
    
    // Test 2: Sequential values
    printf("\nTest 2: Sequential values (0, 1, 2, ...)\n");
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)i;
    }
    
    cpu_result = cpuSumReduction(h_input, size);
    printf("CPU Result: %.2f\n", cpu_result);
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    ConvergentSumReductionKernel<<<1, threadsPerBlock>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("GPU Result (Convergent): %.2f\n", h_output);
    correct = fabs(cpu_result - h_output) < epsilon;
    printf("Status: %s (difference: %.6f)\n", correct ? "PASS" : "FAIL", 
           fabs(cpu_result - h_output));
    
    // Test 3: Random values
    printf("\nTest 3: Random values\n");
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)(rand() % 100) / 10.0f;
    }
    
    cpu_result = cpuSumReduction(h_input, size);
    printf("CPU Result: %.2f\n", cpu_result);
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    ConvergentSumReductionKernel<<<1, threadsPerBlock>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("GPU Result (Convergent): %.2f\n", h_output);
    correct = fabs(cpu_result - h_output) < epsilon;
    printf("Status: %s (difference: %.6f)\n", correct ? "PASS" : "FAIL", 
           fabs(cpu_result - h_output));
    
    printf("\n===============================================\n");
    
    // Test SharedMemorySumReductionKernel
    printf("\n\n========== TESTING SHARED MEMORY KERNEL ==========\n");
    
    // Test 1: All 1s
    printf("\nTest 1: All elements = 1.0\n");
    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f;
    }
    
    cpu_result = cpuSumReduction(h_input, size);
    printf("CPU Result: %.2f\n", cpu_result);
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    SharedMemorySumReductionKernel<<<1, BLOCK_DIM>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("GPU Result (SharedMemory): %.2f\n", h_output);
    correct = fabs(cpu_result - h_output) < epsilon;
    printf("Status: %s (difference: %.6f)\n", correct ? "PASS" : "FAIL", 
           fabs(cpu_result - h_output));
    
    // Test 2: Sequential values
    printf("\nTest 2: Sequential values (0, 1, 2, ...)\n");
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)i;
    }
    
    cpu_result = cpuSumReduction(h_input, size);
    printf("CPU Result: %.2f\n", cpu_result);
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    SharedMemorySumReductionKernel<<<1, BLOCK_DIM>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("GPU Result (SharedMemory): %.2f\n", h_output);
    correct = fabs(cpu_result - h_output) < epsilon;
    printf("Status: %s (difference: %.6f)\n", correct ? "PASS" : "FAIL", 
           fabs(cpu_result - h_output));
    
    // Test 3: Random values
    printf("\nTest 3: Random values\n");
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)(rand() % 100) / 10.0f;
    }
    
    cpu_result = cpuSumReduction(h_input, size);
    printf("CPU Result: %.2f\n", cpu_result);
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    SharedMemorySumReductionKernel<<<1, BLOCK_DIM>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("GPU Result (SharedMemory): %.2f\n", h_output);
    correct = fabs(cpu_result - h_output) < epsilon;
    printf("Status: %s (difference: %.6f)\n", correct ? "PASS" : "FAIL", 
           fabs(cpu_result - h_output));
    
    printf("\n===================================================\n");
    
    // Compare performance between all three kernels
    printf("\n\n========== PERFORMANCE COMPARISON (ALL 3 KERNELS) ==========\n");
    
    // Prepare test data
    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f;
    }
    
    // Timing Simple kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    int numIterations = 1000;
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < numIterations; i++) {
        SimpleSumReductionKernel<<<1, threadsPerBlock>>>(d_input, d_output);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float simpleTime;
    CUDA_CHECK(cudaEventElapsedTime(&simpleTime, start, stop));
    printf("SimpleSumReductionKernel:        %.4f ms (avg over %d runs)\n", 
           simpleTime / numIterations, numIterations);
    
    // Timing Convergent kernel
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < numIterations; i++) {
        ConvergentSumReductionKernel<<<1, threadsPerBlock>>>(d_input, d_output);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float convergentTime;
    CUDA_CHECK(cudaEventElapsedTime(&convergentTime, start, stop));
    printf("ConvergentSumReductionKernel:    %.4f ms (avg over %d runs)\n", 
           convergentTime / numIterations, numIterations);
    
    // Timing SharedMemory kernel
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < numIterations; i++) {
        SharedMemorySumReductionKernel<<<1, BLOCK_DIM>>>(d_input, d_output);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float sharedMemoryTime;
    CUDA_CHECK(cudaEventElapsedTime(&sharedMemoryTime, start, stop));
    printf("SharedMemorySumReductionKernel:  %.4f ms (avg over %d runs)\n", 
           sharedMemoryTime / numIterations, numIterations);
    
    printf("\nPerformance Comparison:\n");
    printf("  Convergent vs Simple:     %.2fx\n", simpleTime / convergentTime);
    printf("  SharedMemory vs Simple:   %.2fx\n", simpleTime / sharedMemoryTime);
    printf("  SharedMemory vs Convergent: %.2fx\n", convergentTime / sharedMemoryTime);
    printf("=============================================================\n");
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_input);
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\nProgram completed successfully!\n");
    return 0;
}
