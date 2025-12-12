#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "cdiv.h"

// __global__ kernel from the attachment (square matrices of size Width)
__global__ void MatrixMulKernel(const float* M, const float* N, float* P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < Width) && (col < Width)) {
        float Pvalue = 0.0f;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = Pvalue;
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
    checkCuda(cudaMalloc(&d_B, bytes), "cudaMalloc d_B");
    checkCuda(cudaMalloc(&d_C, bytes), "cudaMalloc d_C");

    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "memcpy A H2D");
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "memcpy B H2D");

    dim3 block(16, 16);
    dim3 grid(cdiv(Width, block.x), cdiv(Width, block.y));
    MatrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, Width);
    checkCuda(cudaGetLastError(), "Kernel launch");

    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "memcpy C D2H");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Validate against cuBLAS SGEMM: C = A * B
bool validateWithCublas(const float* h_A, const float* h_B, const float* h_C, int Width, float* outMaxAbsDiff) {
    const size_t elems = static_cast<size_t>(Width) * static_cast<size_t>(Width);
    const size_t bytes = elems * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A (cublas)");
    checkCuda(cudaMalloc(&d_B, bytes), "cudaMalloc d_B (cublas)");
    checkCuda(cudaMalloc(&d_C, bytes), "cudaMalloc d_C (cublas)");

    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "H2D A (cublas)");
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "H2D B (cublas)");

    cublasHandle_t handle; 
    checkCublas(cublasCreate(&handle), "cublasCreate");

    // cuBLAS uses column-major. Our host data is row-major.
    // To compute row-major C = A * B, we can compute column-major C^T = B_col * A_col
    // which maps to cublasSgemm with opN/opN and swapped operands (B then A).
    const float alpha = 1.0f;
    const float beta = 0.0f;
    checkCublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N, // B * A
            Width,        // m
            Width,        // n
            Width,        // k
            &alpha,
            d_B, Width,   // A operand (B_col)
            d_A, Width,   // B operand (A_col)
            &beta,
            d_C, Width    // C_col stores C^T
        ), "cublasSgemm"
    );

    std::vector<float> h_C_ref(elems);
    checkCuda(cudaMemcpy(h_C_ref.data(), d_C, bytes, cudaMemcpyDeviceToHost), "D2H C (cublas)");

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    float maxAbsDiff = 0.0f;
    for (size_t i = 0; i < elems; ++i) {
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

    std::vector<float> A(elems), B(elems), C(elems);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < elems; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    launchMatrixMul(A.data(), B.data(), C.data(), Width);

    float maxDiff = 0.0f;
    bool ok = validateWithCublas(A.data(), B.data(), C.data(), Width, &maxDiff);
    std::cout << "Validation vs cuBLAS: " << (ok ? "OK" : "FAIL") << ", maxAbsDiff=" << maxDiff << std::endl;

    return ok ? 0 : 1;
}
