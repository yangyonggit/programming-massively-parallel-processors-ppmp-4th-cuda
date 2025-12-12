#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "cdiv.h"

// __global__ kernel from the attachment (square matrices of size Width)
__global__ void MatrixMulVector(const float* A, const float* B, float* C, int Width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < Width) {      
        float Pvalue = 0.0f;
        for (int k = 0; k < Width; ++k) {
            Pvalue += A[row * Width + k] * B[k];
        }
        C[row] = Pvalue;        
    }
}


static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

static void checkCublas(cublasStatus_t stat, const char* msg) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << msg << " (code " << stat << ")" << std::endl;
        std::exit(1);
    }
}

void launchMatrixMul(const float* h_A, const float* h_B, float* h_C, int Width) {
    const size_t elems = static_cast<size_t>(Width) * static_cast<size_t>(Width);
    const size_t bytes = elems * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");
    checkCuda(cudaMalloc(&d_B, Width * sizeof(float)), "cudaMalloc d_B");
    checkCuda(cudaMalloc(&d_C, Width * sizeof(float)), "cudaMalloc d_C");

    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "memcpy A H2D");
    checkCuda(cudaMemcpy(d_B, h_B, Width * sizeof(float), cudaMemcpyHostToDevice), "memcpy B H2D");

    dim3 block(256,1,1);
    dim3 grid(cdiv(Width, block.x), 1,1);
    MatrixMulVector<<<grid, block>>>(d_A, d_B, d_C, Width);
    checkCuda(cudaGetLastError(), "Kernel launch");

    checkCuda(cudaMemcpy(h_C, d_C, Width * sizeof(float), cudaMemcpyDeviceToHost), "memcpy C D2H");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Validate against cuBLAS SGEMV: C = A * B (matrix-vector multiply)
bool validateWithCublas(cublasHandle_t handle,
                        const float* h_A, const float* h_B, const float* h_C, 
                        int Width, float* outMaxAbsDiff) {
    const size_t matBytes = static_cast<size_t>(Width) * static_cast<size_t>(Width) * sizeof(float);
    const size_t vecBytes = Width * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCuda(cudaMalloc(&d_A, matBytes), "cudaMalloc d_A (cublas)");
    checkCuda(cudaMalloc(&d_B, vecBytes), "cudaMalloc d_B (cublas)");
    checkCuda(cudaMalloc(&d_C, vecBytes), "cudaMalloc d_C (cublas)");

    // Transpose A from row-major (kernel) to column-major (cuBLAS)
    std::vector<float> h_A_colmajor(Width * Width);
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            h_A_colmajor[col * Width + row] = h_A[row * Width + col];
        }
    }

    checkCuda(cudaMemcpy(d_A, h_A_colmajor.data(), matBytes, cudaMemcpyHostToDevice), "H2D A (cublas)");
    checkCuda(cudaMemcpy(d_B, h_B, vecBytes, cudaMemcpyHostToDevice), "H2D B (cublas)");

    // Warm-up GEMV (not timed): avoids first-call initialization overhead
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int incx = 1, incy = 1;
    checkCublas(
        cublasSgemv(
            handle,
            CUBLAS_OP_N,  // no transpose
            Width, Width, // m=Width, n=Width (row-major A treated as Width x Width)
            &alpha,
            d_A, Width,   // A, lda=Width
            d_B, incx,    // x (vector B), increment 1
            &beta,
            d_C, incy     // y (result), increment 1
        ), "cublasSgemv warmup"
    );

    // Timed GEMV
    cudaEvent_t startGemv, stopGemv;
    cudaEventCreate(&startGemv);
    cudaEventCreate(&stopGemv);
    cudaEventRecord(startGemv);
    checkCublas(
        cublasSgemv(
            handle,
            CUBLAS_OP_N,
            Width, Width,
            &alpha,
            d_A, Width,
            d_B, incx,
            &beta,
            d_C, incy
        ), "cublasSgemv timed"
    );
    cudaEventRecord(stopGemv);
    cudaEventSynchronize(stopGemv);
    float msGemv = 0.0f;
    cudaEventElapsedTime(&msGemv, startGemv, stopGemv);

    std::vector<float> h_C_ref(Width);
    checkCuda(cudaMemcpy(h_C_ref.data(), d_C, vecBytes, cudaMemcpyDeviceToHost), "D2H C (cublas)");

    std::cout << "cuBLAS SGEMV time: " << msGemv << " ms" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    float maxAbsDiff = 0.0f;
    for (int i = 0; i < Width; ++i) {
        float diff = std::fabs(h_C_ref[i] - h_C[i]);
        if (diff > maxAbsDiff) maxAbsDiff = diff;
    }
    if (outMaxAbsDiff) *outMaxAbsDiff = maxAbsDiff;

    const float tol = 1e-4f;
    return maxAbsDiff < tol;
}

int main(int argc, char** argv) {
    int Width = 256; // default square size
    if (argc >= 2) {
        Width = std::atoi(argv[1]);
        if (Width <= 0) Width = 256;
    }
    std::cout << "MatrixMul size: " << Width << "x" << Width << std::endl;

    const size_t elems = static_cast<size_t>(Width) * static_cast<size_t>(Width);

    std::vector<float> A(elems), B(Width), C(Width);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < elems; ++i) {
        A[i] = dist(rng);       
    }

    for (size_t i = 0; i < Width; ++i) {
        B[i] = dist(rng);       
    }

    // Create cuBLAS handle once
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate(main)");

    launchMatrixMul(A.data(), B.data(), C.data(), Width);

    float maxDiff = 0.0f;
    bool ok = validateWithCublas(handle, A.data(), B.data(), C.data(), Width, &maxDiff);
    std::cout << "Validation vs cuBLAS: " << (ok ? "OK" : "FAIL") << ", maxAbsDiff=" << maxDiff << std::endl;

    cublasDestroy(handle);
    return ok ? 0 : 1;
}
