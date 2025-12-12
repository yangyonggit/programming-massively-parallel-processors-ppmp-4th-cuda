#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// CUDA kernel for vector addition
__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Device function for vector addition
void vecAddDevice(float* A, float* B, float* C, int n) {
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);
    
    // Allocate device memory
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);
    
    // Copy input vectors from host to device
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vecAddKernel<<<gridSize, BLOCK_SIZE>>>(A_d, B_d, C_d, n);
    
    // Copy result vector from device to host
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

// Host function for vector addition (CPU implementation)
void vecAddHost(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

// Check if two vectors are equal within tolerance
bool checkResult(float* C, float* C_ref, int n, float tolerance) {
    for (int i = 0; i < n; i++) {
        float diff = fabs(C[i] - C_ref[i]);
        if (diff > tolerance) {
            printf("Mismatch at index %d: GPU result = %f, CPU result = %f, diff = %f\n",
                   i, C[i], C_ref[i], diff);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int n = 10000;  // Vector size
    
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    
    printf("Vector Addition: n = %d\n", n);
    printf("=================================\n\n");
    
    int size = n * sizeof(float);
    
    // Allocate host memory
    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);
    float* C_gpu = (float*)malloc(size);
    float* C_cpu = (float*)malloc(size);
    
    if (A == NULL || B == NULL || C_gpu == NULL || C_cpu == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }
    
    // Initialize input vectors
    for (int i = 0; i < n; i++) {
        A[i] = (float)i;
        B[i] = (float)(i * 2);
    }
    
    // Compute on GPU
    printf("Computing on GPU...\n");
    vecAddDevice(A, B, C_gpu, n);
    
    // Compute on CPU
    printf("Computing on CPU...\n");
    vecAddHost(A, B, C_cpu, n);
    
    // Verify results
    printf("Verifying results...\n");
    bool isCorrect = checkResult(C_gpu, C_cpu, n, 1e-5f);
    
    if (isCorrect) {
        printf("✓ Test PASSED: GPU and CPU results match!\n\n");
        
        // Print first 10 and last 10 elements
        printf("First 10 elements:\n");
        for (int i = 0; i < 10 && i < n; i++) {
            printf("A[%d] + B[%d] = %f + %f = %f\n", i, i, A[i], B[i], C_gpu[i]);
        }
        
        if (n > 20) {
            printf("\n...\n\n");
            printf("Last 10 elements:\n");
            for (int i = n - 10; i < n; i++) {
                printf("A[%d] + B[%d] = %f + %f = %f\n", i, i, A[i], B[i], C_gpu[i]);
            }
        }
    } else {
        printf("✗ Test FAILED: Results do not match!\n");
        free(A);
        free(B);
        free(C_gpu);
        free(C_cpu);
        return 1;
    }
    
    // Free host memory
    free(A);
    free(B);
    free(C_gpu);
    free(C_cpu);
    
    printf("\n=================================\n");
    printf("Vector addition completed successfully!\n");
    
    return 0;
}
